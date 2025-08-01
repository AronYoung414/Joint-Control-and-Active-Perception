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

from product_pomdp import prod_pomdp

prod_pomdp = prod_pomdp()


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
    # You may need to modify this based on your observation space
    try:
        return prod_pomdp.observations.index(obs)
    except:
        return 0  # Default to first observation if not found


def pi_theta_network(policy_net, obs_sequence, a, is_last_step=False):
    """
    Get policy probability for action a given observation sequence
    Args:
        policy_net: Policy network
        obs_sequence: List of observations
        a: Action index
        is_last_step: Whether this is the last step (when 'end' action is allowed)
    Returns:
        Probability of taking action a
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


def log_policy_gradient_network(policy_net, obs_sequence, a):
    """
    Compute log policy gradient for neural network
    Args:
        policy_net: Policy network
        obs_sequence: List of observations
        a: Action taken
    Returns:
        Gradient of log policy
    """
    # Convert observations to indices
    obs_indices = [obs_to_index(obs) for obs in obs_sequence]
    obs_tensor = torch.tensor([obs_indices], dtype=torch.long)

    # Forward pass
    action_probs, action_logits = policy_net(obs_tensor)

    # Compute log probability of taken action
    log_prob = torch.log(action_probs[0, a] + 1e-8)  # Add small epsilon for numerical stability

    # Compute gradients
    log_prob.backward()

    # Extract gradients (you might want to return them in a different format)
    gradients = []
    for param in policy_net.parameters():
        if param.grad is not None:
            gradients.append(param.grad.clone())

    return gradients


class LazyObservableOperator:
    """
    Lazy computation with caching - only compute what you need, when you need it
    This is often the best approach for large state spaces
    """

    def __init__(self):
        self.cache = {}
        self.state_to_idx = {state: i for i, state in enumerate(prod_pomdp.states)}

    def get_operator(self, obs_t, act_t):
        """Compute observable operator on-demand and cache results"""
        key = (obs_t, act_t)
        if key not in self.cache:
            rows, cols, data = [], [], []

            # Only iterate over states that have transitions for this action
            for st_prime in prod_pomdp.states:
                if (st_prime not in prod_pomdp.transition or
                        act_t not in prod_pomdp.transition[st_prime]):
                    continue

                if (st_prime not in prod_pomdp.emiss or
                        act_t not in prod_pomdp.emiss[st_prime] or
                        obs_t not in prod_pomdp.emiss[st_prime][act_t]):
                    continue

                j = self.state_to_idx[st_prime]
                emiss_prob = prod_pomdp.emiss[st_prime][act_t][obs_t]

                if emiss_prob == 0:
                    continue

                # Only process reachable next states
                for st in prod_pomdp.transition[st_prime][act_t]:
                    trans_prob = prod_pomdp.transition[st_prime][act_t][st]
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
                                shape=(prod_pomdp.state_size, prod_pomdp.state_size))
            else:
                oo = csr_matrix((prod_pomdp.state_size, prod_pomdp.state_size))

            self.cache[key] = oo

        return self.cache[key]


def get_observable_operator_ultra_fast():
    """
    Ultra-fast implementation using vectorized operations and minimal memory
    """
    # Use lazy computation instead of precomputing everything
    return LazyObservableOperator()


def get_observable_operator():
    """
    Highly optimized sparse implementation for large state spaces
    Avoids nested loops by only processing valid transitions
    """
    oo_dict = {}

    # Pre-compute state index mapping for faster lookups
    state_to_idx = {state: i for i, state in enumerate(prod_pomdp.states)}

    for obs_t in prod_pomdp.observations:
        oo_dict[obs_t] = {}
        for act_t in prod_pomdp.actions:
            rows, cols, data = [], [], []

            # Only iterate over states that actually have transitions for this action
            for st_prime in prod_pomdp.states:
                # Skip if this state-action pair has no transitions
                if (st_prime not in prod_pomdp.transition or
                        act_t not in prod_pomdp.transition[st_prime]):
                    continue

                # Skip if this state-action pair has no emission for this observation
                if (st_prime not in prod_pomdp.emiss or
                        act_t not in prod_pomdp.emiss[st_prime] or
                        obs_t not in prod_pomdp.emiss[st_prime][act_t]):
                    continue

                j = state_to_idx[st_prime]
                emiss_prob = prod_pomdp.emiss[st_prime][act_t][obs_t]

                # Skip if emission probability is zero
                if emiss_prob == 0:
                    continue

                # Only process reachable next states
                for st in prod_pomdp.transition[st_prime][act_t]:
                    trans_prob = prod_pomdp.transition[st_prime][act_t][st]
                    if trans_prob > 0:  # Only non-zero transitions
                        i = state_to_idx[st]
                        value = trans_prob * emiss_prob
                        if value > 0:
                            rows.append(i)
                            cols.append(j)
                            data.append(value)

            # Create sparse matrix only if there are non-zero entries
            if data:
                oo_dict[obs_t][act_t] = csr_matrix(
                    (data, (rows, cols)),
                    shape=(prod_pomdp.state_size, prod_pomdp.state_size)
                )
            else:
                # Empty sparse matrix
                oo_dict[obs_t][act_t] = csr_matrix((prod_pomdp.state_size, prod_pomdp.state_size))

    return oo_dict


def p_obs_g_actions(y, a_list, observable_operator):
    """
    Modified to work efficiently with sparse observable operators
    Supports both dict-based and lazy operator implementations
    """
    act_list = [prod_pomdp.actions[a] for a in a_list]
    mu_0 = prod_pomdp.initial_dist

    # Handle both dict-based and lazy operator implementations
    if hasattr(observable_operator, 'get_operator'):
        # Lazy operator
        oo = observable_operator.get_operator(y[-1], act_list[-1])
    else:
        # Dict-based operator
        oo = observable_operator[y[-1]][act_list[-1]]

    # Use sparse matrix operations - convert mu_0 to proper shape if needed
    one_vec = np.ones((1, prod_pomdp.state_size))
    probs = one_vec @ oo  # Sparse matrix multiplication

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
    Supports both dict-based and lazy operator implementations
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
    Supports both dict-based and lazy operator implementations
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


def log_p_theta_obs_network(policy_net, y, a_list, observable_operator):
    """
    Compute log P(y, a_list; theta) for neural network policy
    """
    # For the first action, use empty observation sequence
    policy_sum = np.log2(pi_theta_network(policy_net, [], a_list[0], is_last_step=False))

    # For subsequent actions, use observation history
    for i in range(len(y) - 1):
        obs_history = y[:i + 1]
        is_last = (i == len(y) - 2)  # Check if this is the last action
        policy_sum += np.log2(pi_theta_network(policy_net, obs_history, a_list[i + 1], is_last_step=is_last))

    p_obs_g_acts_initial = p_obs_g_actions_initial(y[0], a_list[0], observable_operator)
    p_obs_g_acts = p_obs_g_actions(y, a_list, observable_operator)
    log_p_y_g_sas0 = np.log2(p_obs_g_acts) if p_obs_g_acts > 0 else float('-inf')

    return log_p_y_g_sas0 - np.log2(p_obs_g_acts_initial) + policy_sum


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
                # End action at last step has probability 1, so log(1) = 0
                # No gradient contribution
                continue
            else:
                # End action before last step has probability 0
                # This should not happen in valid trajectories
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
            # If no gradient, create zero gradient with same shape
            gradients.append(torch.zeros_like(param))

    return gradients


def p_zT_g_y(y, a_list, observable_operator):
    p_obs_g_acts = p_obs_g_actions(y, a_list, observable_operator)
    act_T = prod_pomdp.actions[a_list[-1]]
    o_T = y[-1]
    p_zT1 = 0
    for st_T in prod_pomdp.secret_states:
        v_T = prod_pomdp.states.index(st_T)
        emiss_prob = prod_pomdp.emiss[st_T][act_T][o_T]
        p_vtp1_obs_g_acts = p_vtp1_obs_g_actions(v_T, y, a_list, observable_operator)
        p_zT1 += emiss_prob * p_vtp1_obs_g_acts / p_obs_g_acts
    p_zT0 = 1 - p_zT1
    p_wT1 = 0
    for st_T in prod_pomdp.goal_states:
        v_T = prod_pomdp.states.index(st_T)
        emiss_prob = prod_pomdp.emiss[st_T][act_T][o_T]
        p_vtp1_obs_g_acts = p_vtp1_obs_g_actions(v_T, y, a_list, observable_operator)
        p_wT1 += emiss_prob * p_vtp1_obs_g_acts / p_obs_g_acts
    return p_zT1, p_zT0, p_wT1


def posterior(y, a_list, observable_operator):
    p_obs_g_acts = p_obs_g_actions(y, a_list, observable_operator)
    act_T = prod_pomdp.actions[a_list[-1]]
    o_T = y[-1]
    p_zT1 = 0
    for st_T in prod_pomdp.secret_states:
        v_T = prod_pomdp.states.index(st_T)
        emiss_prob = prod_pomdp.emiss[st_T][act_T][o_T]
        p_vtp1_obs_g_acts = p_vtp1_obs_g_actions(v_T, y[0:-1], a_list[0:-1], observable_operator)
        p_zT1 += emiss_prob * p_vtp1_obs_g_acts / p_obs_g_acts
    p_zT0 = 1 - p_zT1
    return p_zT1, p_zT0


def entropy_a_grad_network(policy_net, s_data, y_data, a_data, observable_operator):
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

        states = [prod_pomdp.states[s] for s in s_data[k]]
        # print(states)
        # print(y_k)

        # Get gradients for this trajectory
        grad_log_P_theta_yk = nabla_log_p_theta_obs_network(policy_net, y_k, a_list_k)

        p_zT1, p_zT0, p_wT1 = p_zT_g_y(y_k, a_list_k, observable_operator)

        # print("The probability of being adversry", p_zT1)
        # print("The probability of completing task", p_wT1)
        # print('#' * 100)

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
    Args:
        policy_net: Policy network
        obs_sequence: List of observations
        is_last_step: Whether this is the last step (when 'end' action should be chosen)
    Returns:
        Sampled action
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


def display_sampled_data(s_data, y_data, a_data, num_trajectories_to_show=5):
    """
    Display sampled states, observations, and actions for debugging
    Args:
        s_data: Array of state indices
        y_data: List of observation sequences
        a_data: List of action sequences
        num_trajectories_to_show: Number of trajectories to display
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

        # Display step-by-step breakdown
        # print("Step-by-step breakdown:")
        # for i in range(len(y_k)):
        #     if i < len(a_list_k):
        #         if i == 0:
        #             print(f"  Step {i}: Action={action_names[i]} -> Observation={y_k[i]}")
        #         else:
        #             print(f"  Step {i}: Observation={y_k[i - 1]} -> Action={action_names[i]} -> Observation={y_k[i]}")
        #
        # print("-" * 30)

    print(f"{'=' * 50}\n")


def display_states_from_s_data(s_data, num_trajectories_to_show=5):
    """
    Display state sequences from s_data
    """
    M = min(len(s_data), num_trajectories_to_show)
    print(f"\nSTATE SEQUENCES (showing {M} trajectories):")
    for k in range(M):
        s_list = s_data[k]
        state_names = [prod_pomdp.states[s] for s in s_list if s < len(prod_pomdp.states)]
        print(f"Trajectory {k + 1}: {state_names}")
    print()


def sample_data_network(policy_net, M, T, type='mix'):
    """
    Sample data using neural network policy
    Ensures that 'end' action is only selected at the last step
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

        # Get the observation of initial state
        obs = prod_pomdp.observation_function_sampler(state, act)
        y.append(obs)

        # Sample intermediate actions (T-1 steps)
        for t in range(T - 1):
            s = prod_pomdp.states.index(state)
            s_data[m, t] = s

            # Sample the next state
            state = prod_pomdp.next_state_sampler(state, act)

            # Sample sensing action using observation history (cannot be 'end' action)
            act = action_sampler_network(policy_net, y, is_last_step=False)
            a = prod_pomdp.actions.index(act)
            a_list.append(a)

            # Add the observation
            obs = prod_pomdp.observation_function_sampler(state, act)
            y.append(obs)

        # The last action must be 'end' action
        act = action_sampler_network(policy_net, y, is_last_step=True)  # This will return 'e'
        a = prod_pomdp.actions.index(act)
        a_list.append(a)

        # Final step: transition to final state and add end action
        state = prod_pomdp.next_state_sampler(state, act)
        s = prod_pomdp.states.index(state)
        s_data[m, T - 1] = s

        # Get final observation
        obs = prod_pomdp.observation_function_sampler(state, act)
        y.append(obs)

        # Process the data
        y_data.append(y)
        a_data.append(a_list)

    return s_data, y_data, a_data


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
    ex_num = 11
    iter_num = 2000
    M = 100
    T = 8
    eta = 0.001  # Learning rate for neural network (typically smaller)
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

    # Alternative: Use precomputed sparse matrices (uncomment if you prefer)
    # print("Computing sparse observable operators...")
    # obs_dict = get_observable_operator()

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
            policy_net, s_data, y_data, a_data, obs_dict
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
