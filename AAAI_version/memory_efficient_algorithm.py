import numpy as np
import time
import matplotlib.pyplot as plt
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from scipy.sparse import csr_matrix, lil_matrix

from random import choices
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat

# Import the memory-efficient classes
from grid_world_example_efficient import Environment
from pomdp_grid_efficient import POMDP
from DFA import DFA
from product_pomdp_efficient import ProductPOMDP

# Create the memory-efficient environment
grid_world = Environment(width=10, height=10)  # Can now handle larger grids
pomdp = POMDP(grid_world)

# Create DFA and Product POMDP
dfa = DFA(pomdp)
prod_pomdp = ProductPOMDP(pomdp, dfa)


class PolicyNetwork(nn.Module):
    def __init__(self, obs_vocab_size, action_size, hidden_dim=64, max_seq_len=20):
        super(PolicyNetwork, self).__init__()
        self.obs_vocab_size = obs_vocab_size
        self.action_size = action_size - 1  # Exclude 'end' action from policy network
        self.hidden_dim = hidden_dim
        self.max_seq_len = max_seq_len

        # Embedding layer for observations
        self.obs_embedding = nn.Embedding(obs_vocab_size, hidden_dim)

        # LSTM to process observation sequences
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)

        # Output layer to produce action probabilities (only for non-end actions)
        self.action_head = nn.Linear(hidden_dim, self.action_size)

    def forward(self, obs_sequences, sequence_lengths=None):
        """
        Forward pass of the policy network
        Args:
            obs_sequences: Tensor of shape (batch_size, seq_len) containing observation indices
            sequence_lengths: Tensor of shape (batch_size,) containing actual sequence lengths
        Returns:
            action_probs: Tensor of shape (batch_size, action_size) containing action probabilities
        """
        batch_size, seq_len = obs_sequences.shape

        # Embed observations
        embedded_obs = self.obs_embedding(obs_sequences)  # (batch_size, seq_len, hidden_dim)

        # Process through LSTM
        if sequence_lengths is not None:
            # Pack padded sequences for efficiency
            packed_embedded = nn.utils.rnn.pack_padded_sequence(
                embedded_obs, sequence_lengths, batch_first=True, enforce_sorted=False
            )
            lstm_out, (hidden, cell) = self.lstm(packed_embedded)
            # Use the last hidden state
            final_hidden = hidden[-1]  # (batch_size, hidden_dim)
        else:
            lstm_out, (hidden, cell) = self.lstm(embedded_obs)
            final_hidden = hidden[-1]  # (batch_size, hidden_dim)

        # Generate action probabilities
        action_logits = self.action_head(final_hidden)
        action_probs = F.softmax(action_logits, dim=-1)

        return action_probs, action_logits


def obs_to_index(obs):
    """Convert observation to index for embedding"""
    try:
        return prod_pomdp.observations.index(obs)
    except:
        return 0  # Default to first observation if not found


def pi_theta_network(policy_net, obs_sequence, a, is_last_step=False):
    """
    Get policy probability for action a given observation sequence
    """
    # Get the end action index
    end_action_idx = len(prod_pomdp.actions) - 1  # Assuming 'end' is the last action

    # Handle end action case
    if a == end_action_idx:
        if is_last_step:
            return 1.0  # End action has probability 1 at the last step
        else:
            return 0.0  # End action is not allowed before the last step

    # Handle regular actions (non-end actions)
    if a >= policy_net.action_size:
        return 0.0  # Invalid action index

    # Convert observations to indices
    obs_indices = [obs_to_index(obs) for obs in obs_sequence]
    obs_tensor = torch.tensor([obs_indices], dtype=torch.long)

    with torch.no_grad():
        action_probs, _ = policy_net(obs_tensor)
        return action_probs[0, a].item()


class LazyObservableOperator:
    """
    Memory-efficient observable operator that computes values on-the-fly
    """

    def __init__(self, prod_pomdp):
        self.prod_pomdp = prod_pomdp
        self.cache = {}
        self.state_to_idx = {state: i for i, state in enumerate(prod_pomdp.states)}

    def get_operator(self, obs_t, act_t):
        """Compute observable operator on-demand and cache results"""
        key = (obs_t, act_t)
        if key not in self.cache:
            rows, cols, data = [], [], []

            # Iterate over all states
            for st_prime in self.prod_pomdp.states:
                j = self.state_to_idx[st_prime]

                # Get emission probability using the memory-efficient method
                emiss_prob = self.prod_pomdp.get_emission_prob(st_prime, act_t, obs_t)

                if emiss_prob == 0:
                    continue

                # Get all possible next states using memory-efficient method
                possible_next_states = self.prod_pomdp.get_possible_next_states(st_prime, act_t)

                for st in possible_next_states:
                    # Get transition probability using memory-efficient method
                    trans_prob = self.prod_pomdp.get_transition_prob(st_prime, act_t, st)
                    if trans_prob > 0:
                        i = self.state_to_idx[st]
                        value = trans_prob * emiss_prob
                        if value > 0:
                            rows.append(i)
                            cols.append(j)
                            data.append(value)

            # Create sparse matrix
            if data:
                oo = csr_matrix((data, (rows, cols)),
                                shape=(self.prod_pomdp.state_size, self.prod_pomdp.state_size))
            else:
                oo = csr_matrix((self.prod_pomdp.state_size, self.prod_pomdp.state_size))

            self.cache[key] = oo

        return self.cache[key]


def get_observable_operator_ultra_fast():
    """
    Ultra-fast implementation using vectorized operations and minimal memory
    """
    return LazyObservableOperator(prod_pomdp)


def get_observable_operator():
    """
    Memory-efficient sparse implementation for large state spaces
    """
    oo_dict = {}
    state_to_idx = {state: i for i, state in enumerate(prod_pomdp.states)}

    for obs_t in prod_pomdp.observations:
        oo_dict[obs_t] = {}
        for act_t in prod_pomdp.actions:
            rows, cols, data = [], [], []

            for st_prime in prod_pomdp.states:
                j = state_to_idx[st_prime]

                # Use memory-efficient emission probability computation
                emiss_prob = prod_pomdp.get_emission_prob(st_prime, act_t, obs_t)

                if emiss_prob == 0:
                    continue

                # Use memory-efficient transition computation
                possible_next_states = prod_pomdp.get_possible_next_states(st_prime, act_t)

                for st in possible_next_states:
                    trans_prob = prod_pomdp.get_transition_prob(st_prime, act_t, st)
                    if trans_prob > 0:
                        i = state_to_idx[st]
                        value = trans_prob * emiss_prob
                        if value > 0:
                            rows.append(i)
                            cols.append(j)
                            data.append(value)

            # Create sparse matrix
            if data:
                oo_dict[obs_t][act_t] = csr_matrix(
                    (data, (rows, cols)),
                    shape=(prod_pomdp.state_size, prod_pomdp.state_size)
                )
            else:
                oo_dict[obs_t][act_t] = csr_matrix((prod_pomdp.state_size, prod_pomdp.state_size))

    return oo_dict


def p_obs_g_actions(y, a_list, observable_operator):
    """
    Modified to work efficiently with sparse observable operators
    """
    act_list = [prod_pomdp.actions[a] for a in a_list]
    mu_0 = prod_pomdp.initial_dist

    # Handle both dict-based and lazy operator implementations
    if hasattr(observable_operator, 'get_operator'):
        oo = observable_operator.get_operator(y[-1], act_list[-1])
    else:
        oo = observable_operator[y[-1]][act_list[-1]]

    # Use sparse matrix operations
    one_vec = np.ones((1, prod_pomdp.state_size))
    probs = one_vec @ oo

    # Calculate the probability using sparse operations
    for t in reversed(range(len(y) - 1)):
        if hasattr(observable_operator, 'get_operator'):
            oo = observable_operator.get_operator(y[t], act_list[t])
        else:
            oo = observable_operator[y[t]][act_list[t]]
        probs = probs @ oo

    # Final multiplication with initial distribution
    probs = probs @ mu_0

    # Handle both sparse and dense result formats
    if hasattr(probs, 'toarray'):
        return probs.toarray()[0, 0]
    elif hasattr(probs, 'shape') and len(probs.shape) > 0:
        return probs[0] if probs.shape == (1,) else probs[0, 0]
    else:
        return float(probs)


def p_obs_g_actions_initial(o_0, a_0, observable_operator):
    """
    Modified to work efficiently with sparse observable operators
    """
    act_0 = prod_pomdp.actions[a_0]
    mu_0 = prod_pomdp.initial_dist

    # Handle both dict-based and lazy operator implementations
    if hasattr(observable_operator, 'get_operator'):
        oo = observable_operator.get_operator(o_0, act_0)
    else:
        oo = observable_operator[o_0][act_0]

    # Use sparse matrix operations
    one_vec = np.ones((1, prod_pomdp.state_size))
    probs = one_vec @ oo
    probs = probs @ mu_0

    # Handle both sparse and dense result formats
    if hasattr(probs, 'toarray'):
        return probs.toarray()[0, 0]
    elif hasattr(probs, 'shape') and len(probs.shape) > 0:
        return probs[0] if probs.shape == (1,) else probs[0, 0]
    else:
        return float(probs)


def p_vtp1_obs_g_actions(v_tp1, y, a_list, observable_operator):
    """
    Modified to work efficiently with sparse observable operators
    """
    act_list = [prod_pomdp.actions[a] for a in a_list]
    mu_0 = prod_pomdp.initial_dist

    # Handle both dict-based and lazy operator implementations
    if hasattr(observable_operator, 'get_operator'):
        oo = observable_operator.get_operator(y[-1], act_list[-1])
    else:
        oo = observable_operator[y[-1]][act_list[-1]]

    # Create a one-hot vector which has a 1 element at state index v_{t+1}
    one_hot = np.zeros((1, prod_pomdp.state_size))
    one_hot[0, v_tp1] = 1

    # Use sparse matrix operations
    probs = one_hot @ oo

    # Calculate the probability using sparse operations
    for t in reversed(range(len(y) - 1)):
        if hasattr(observable_operator, 'get_operator'):
            oo = observable_operator.get_operator(y[t], act_list[t])
        else:
            oo = observable_operator[y[t]][act_list[t]]
        probs = probs @ oo

    # Final multiplication with initial distribution
    probs = probs @ mu_0

    # Handle both sparse and dense result formats
    if hasattr(probs, 'toarray'):
        return probs.toarray()[0, 0]
    elif hasattr(probs, 'shape') and len(probs.shape) > 0:
        return probs[0] if probs.shape == (1,) else probs[0, 0]
    else:
        return float(probs)


def p_theta_obs_network(policy_net, y, a_list, observable_operator):
    """
    Compute P(y, a_list; theta) for neural network policy
    """
    # For the first action, use empty observation sequence
    policy_prod = pi_theta_network(policy_net, [], a_list[0], is_last_step=False)

    # For subsequent actions, use observation history
    for i in range(len(y) - 1):
        obs_history = y[:i + 1]
        is_last = (i == len(y) - 2)  # Check if this is the last action
        policy_prod *= pi_theta_network(policy_net, obs_history, a_list[i + 1], is_last_step=is_last)

    p_obs_g_acts_initial = p_obs_g_actions_initial(y[0], a_list[0], observable_operator)
    p_obs_g_acts = p_obs_g_actions(y, a_list, observable_operator)

    return p_obs_g_acts / p_obs_g_acts_initial * policy_prod


def nabla_log_p_theta_obs_network(policy_net, y, a_list):
    """
    Compute gradient of log P(y, a_list; theta) for neural network policy
    """
    policy_net.zero_grad()
    total_log_prob = 0

    # Get the end action index
    end_action_idx = len(prod_pomdp.actions) - 1

    # For the first action, use empty observation sequence
    if a_list[0] != end_action_idx:  # Skip end action (no gradient)
        obs_tensor = torch.zeros((1, 1), dtype=torch.long)  # Use dummy observation for empty sequence
        action_probs, _ = policy_net(obs_tensor)
        log_prob = torch.log(action_probs[0, a_list[0]] + 1e-8)
        total_log_prob += log_prob

    # For subsequent actions, use observation history
    for i in range(len(y) - 1):
        is_last_action = (i == len(y) - 2)  # Check if this is the last action
        current_action = a_list[i + 1]

        # Skip end action unless it's the last step
        if current_action == end_action_idx:
            if is_last_action:
                continue
            else:
                continue

        # Process regular actions
        if current_action < policy_net.action_size:
            obs_history = y[:i + 1]
            obs_indices = [obs_to_index(obs) for obs in obs_history]
            obs_tensor = torch.tensor([obs_indices], dtype=torch.long)

            action_probs, _ = policy_net(obs_tensor)
            log_prob = torch.log(action_probs[0, current_action] + 1e-8)
            total_log_prob += log_prob

    # Only compute gradients if we have valid log probabilities
    if total_log_prob != 0:
        total_log_prob.backward()

    # Extract gradients
    gradients = []
    for param in policy_net.parameters():
        if param.grad is not None:
            gradients.append(param.grad.clone())
        else:
            gradients.append(torch.zeros_like(param))

    return gradients


def p_zT_g_y(y, a_list, observable_operator):
    p_obs_g_acts = p_obs_g_actions(y, a_list, observable_operator)
    act_T = prod_pomdp.actions[a_list[-1]]
    o_T = y[-1]
    p_zT1 = 0
    for st_T in prod_pomdp.secret_states:
        v_T = prod_pomdp.states.index(st_T)
        # Use memory-efficient emission probability computation
        emiss_prob = prod_pomdp.get_emission_prob(st_T, act_T, o_T)
        p_vtp1_obs_g_acts = p_vtp1_obs_g_actions(v_T, y[0:-1], a_list[0:-1], observable_operator)
        p_zT1 += emiss_prob * p_vtp1_obs_g_acts / p_obs_g_acts
    p_zT0 = 1 - p_zT1
    p_wT1 = 0
    for st_T in prod_pomdp.goal_states:
        v_T = prod_pomdp.states.index(st_T)
        # Use memory-efficient emission probability computation
        emiss_prob = prod_pomdp.get_emission_prob(st_T, act_T, o_T)
        p_vtp1_obs_g_acts = p_vtp1_obs_g_actions(v_T, y[0:-1], a_list[0:-1], observable_operator)
        p_wT1 += emiss_prob * p_vtp1_obs_g_acts / p_obs_g_acts
    return p_zT1, p_zT0, p_wT1


def entropy_a_grad_network(policy_net, y_data, a_data, observable_operator):
    """
    Compute entropy and gradients for neural network policy
    """
    M = len(y_data)
    H = 0
    P = 0

    # Accumulate gradients
    total_grad_H = None
    total_grad_P = None

    for k in range(M):
        y_k = y_data[k]
        a_list_k = a_data[k]

        # Get gradients for this trajectory
        grad_log_P_theta_yk = nabla_log_p_theta_obs_network(policy_net, y_k, a_list_k)

        p_zT1, p_zT0, p_wT1 = p_zT_g_y(y_k, a_list_k, observable_operator)
        print(p_zT1)

        temp_H_1 = p_zT1 * np.log2(p_zT1) if p_zT1 > 0 else 0
        temp_H_0 = p_zT0 * np.log2(p_zT0) if p_zT0 > 0 else 0
        temp_H = temp_H_1 + temp_H_0
        H += temp_H
        P += p_wT1

        # Accumulate gradients
        if total_grad_H is None:
            total_grad_H = [temp_H * grad.clone() for grad in grad_log_P_theta_yk]
            total_grad_P = [p_wT1 * grad.clone() for grad in grad_log_P_theta_yk]
        else:
            for i, grad in enumerate(grad_log_P_theta_yk):
                total_grad_H[i] += temp_H * grad
                total_grad_P[i] += p_wT1 * grad

    H = -H / M
    P = P / M

    # Average gradients
    nabla_H = [-grad / M for grad in total_grad_H]
    nabla_P = [grad / M for grad in total_grad_P]

    return H, nabla_H, P, nabla_P


def action_sampler_network(policy_net, obs_sequence, is_last_step=False):
    """
    Sample action from neural network policy
    """
    # If it's the last step, always return the end action
    if is_last_step:
        return 'e'  # Return the end action directly

    # For non-last steps, sample from the policy network (excluding end action)
    obs_indices = [obs_to_index(obs) for obs in obs_sequence]
    if len(obs_indices) == 0:
        obs_tensor = torch.zeros((1, 1), dtype=torch.long)  # Dummy observation for empty sequence
    else:
        obs_tensor = torch.tensor([obs_indices], dtype=torch.long)

    with torch.no_grad():
        action_probs, _ = policy_net(obs_tensor)
        action_probs_np = action_probs[0].numpy()

    # Sample action from available non-end actions
    action_idx = np.random.choice(len(action_probs_np), p=action_probs_np)
    return prod_pomdp.actions[action_idx]


def sample_data_network(policy_net, M, T, type='mix'):
    """
    Sample data using neural network policy with memory-efficient environment
    """
    s_data = np.zeros([M, T], dtype=np.int32)
    a_data = []
    y_data = []

    for m in range(M):
        y = []
        a_list = []

        # Start from initial state
        if type == 'mix':
            state = choices(prod_pomdp.initial_states, prod_pomdp.initial_dist_sampling, k=1)[0]
        elif type == '1':
            state = prod_pomdp.initial_states[1]
        elif type == '0':
            state = prod_pomdp.initial_states[0]
        else:
            raise ValueError('Invalid type value.')

        # Sample first sensing action (cannot be 'end' action)
        act = action_sampler_network(policy_net, [], is_last_step=False)
        a = prod_pomdp.actions.index(act)
        a_list.append(a)

        # Get the observation of initial state using memory-efficient method
        obs = prod_pomdp.sample_observation(state, act)
        y.append(obs)

        # Sample intermediate actions (T-1 steps)
        for t in range(T - 1):
            s = prod_pomdp.states.index(state)
            s_data[m, t] = s

            # Sample the next state using memory-efficient method
            state = prod_pomdp.sample_next_state(state, act)

            # Sample sensing action using observation history (cannot be 'end' action)
            act = action_sampler_network(policy_net, y, is_last_step=False)
            a = prod_pomdp.actions.index(act)
            a_list.append(a)

            # Add the observation using memory-efficient method
            obs = prod_pomdp.sample_observation(state, act)
            y.append(obs)

        # The last action must be 'end' action
        act = action_sampler_network(policy_net, y, is_last_step=True)  # This will return 'e'
        a = prod_pomdp.actions.index(act)
        a_list.append(a)

        # Final step: transition to final state and add end action
        state = prod_pomdp.sample_next_state(state, act)
        s = prod_pomdp.states.index(state)
        s_data[m, T - 1] = s

        # Get final observation using memory-efficient method
        obs = prod_pomdp.sample_observation(state, act)
        y.append(obs)

        # Process the data
        y_data.append(y)
        a_data.append(a_list)

    return s_data, y_data, a_data


def display_sampled_data(s_data, y_data, a_data, num_trajectories_to_show=5):
    """
    Display sampled states, observations, and actions for debugging
    """
    M = min(len(y_data), num_trajectories_to_show)

    print(f"\n{'=' * 50}")
    print(f"DISPLAYING {M} SAMPLE TRAJECTORIES")
    print(f"{'=' * 50}")

    for k in range(M):
        print(f"\n--- Trajectory {k + 1} ---")

        # Display states (for non-final steps)
        if k < len(s_data):
            state_names = [prod_pomdp.states[s] for s in s_data[k] if s < len(prod_pomdp.states)]
            print(f"States (steps 0-{len(state_names) - 1}): {state_names}")

        # Display observations
        y_k = y_data[k]
        print(f"Observations: {y_k}")

        # Display actions with names
        a_list_k = a_data[k]
        action_names = []
        for a_idx in a_list_k:
            if a_idx < len(prod_pomdp.actions):
                action_names.append(prod_pomdp.actions[a_idx])
            else:
                action_names.append(f"INVALID_ACTION_{a_idx}")
        print(f"Actions: {action_names}")
        print(f"Action indices: {a_list_k}")

    print(f"{'=' * 50}\n")


def apply_gradients(optimizer, gradients):
    """
    Apply computed gradients to the policy network
    """
    for param, grad in zip(optimizer.param_groups[0]['params'], gradients):
        if param.grad is None:
            param.grad = grad
        else:
            param.grad += grad


def main():
    # Define hyperparameters
    ex_num = 1
    iter_num = 1000
    M = 10
    T = 20
    eta = 0.001  # Learning rate for neural network
    alpha = 1

    # Initialize policy network
    obs_vocab_size = len(prod_pomdp.observations)
    action_size = prod_pomdp.action_size
    hidden_dim = 64

    policy_net = PolicyNetwork(obs_vocab_size, action_size, hidden_dim, max_seq_len=T)
    optimizer = optim.Adam(policy_net.parameters(), lr=eta)

    # Use lazy computation for maximum efficiency with large state spaces
    print("Initializing lazy observable operator for efficient computation...")
    obs_dict = get_observable_operator_ultra_fast()  # Uses LazyObservableOperator

    # Create empty lists
    entropy_list = []
    probs_list = []

    # Training loop
    for i in range(iter_num):
        start = time.time()

        # Sample trajectories
        s_data, y_data, a_data = sample_data_network(policy_net, M, T)

        # Compute gradients
        approx_entropy, grad_H, approx_P_Z1, grad_P = entropy_a_grad_network(
            policy_net, y_data, a_data, obs_dict
        )

        # display_sampled_data(s_data, y_data, a_data, num_trajectories_to_show=10)

        # Combine gradients
        combined_gradients = [grad_h - alpha * grad_p for grad_h, grad_p in zip(grad_H, grad_P)]

        print("The conditional entropy is", approx_entropy)
        entropy_list.append(approx_entropy)
        print("The probability of completing the task is", approx_P_Z1)
        probs_list.append(approx_P_Z1)

        # Apply gradients
        optimizer.zero_grad()
        apply_gradients(optimizer, combined_gradients)
        optimizer.step()

        end = time.time()
        print(f"iteration_{i + 1} done. It takes", end - start, "s")
        print("#" * 100)

    # Save results
    with open(f'./data/Values/policy_net_{ex_num}.pkl', "wb") as pkl_wb_obj:
        torch.save(policy_net.state_dict(), pkl_wb_obj)

    with open(f'./data/Values/PZList_{ex_num}', "wb") as pkl_wb_obj:
        pickle.dump(probs_list, pkl_wb_obj)

    with open(f'./data/Values/entropy_{ex_num}', "wb") as pkl_wb_obj:
        pickle.dump(entropy_list, pkl_wb_obj)

    # Plot results
    iteration_list = range(iter_num)
    plt.plot(iteration_list, entropy_list, label=r'entropy $H(Z_T|Y;\theta)$')
    plt.plot(iteration_list, probs_list, label=r'probability $P_\theta(Z_T = 1)$')
    plt.xlabel("The iteration number")
    plt.ylabel("Values")
    plt.legend()
    plt.savefig(f'./data/Graphs/Ex_{ex_num}_network.png')
    plt.show()


if __name__ == "__main__":
    main()
