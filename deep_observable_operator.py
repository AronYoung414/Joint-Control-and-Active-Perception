import itertools
import numpy as np
import time
import matplotlib.pyplot as plt
import pickle
from math import isinf

from random import choices, choice
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat

from product_pomdp import prod_pomdp
from finite_state_controller import FSC
from policy_network import *

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

prod_pomdp = prod_pomdp()
fsc = FSC()


# PRIOR = env.get_prior_distribution([0.1, 0.4, 0.5])
# PRIOR = torch.from_numpy(PRIOR).type(dtype=torch.float32)
# PRIOR = PRIOR.to(device)


def observable_operator(obs_t, act_t):
    oo = np.zeros([prod_pomdp.state_size, prod_pomdp.state_size])
    for i in range(prod_pomdp.state_size):
        for j in range(prod_pomdp.state_size):
            st = prod_pomdp.states[i]
            st_prime = prod_pomdp.states[j]
            # The definition of observable operators
            # print(st_prime, act_t)
            if st in prod_pomdp.next_supp[st_prime][act_t]:
                oo[i, j] = prod_pomdp.transition[st_prime][act_t][st] * prod_pomdp.emiss[st_prime][act_t][obs_t]
    return oo


def p_obs_g_actions(y, a_list):
    # Get the real sensing actions
    act_list = [prod_pomdp.actions[a] for a in a_list]
    # Give value to the initial state
    mu_0 = prod_pomdp.initial_dist
    # Obtain observable operators
    oo = observable_operator(y[-1], act_list[-1])
    # Create a vector with all elements equals to 1
    one_vec = np.ones([1, prod_pomdp.state_size])
    # Initialize the probability of observation given sensing actions and initial states
    probs = one_vec @ oo
    # Calculate the probability
    for t in reversed(range(len(y) - 1)):
        oo = observable_operator(y[t], act_list[t])
        probs = probs @ oo
    # print(y)
    # print(senAct_list)
    # print(probs)
    probs = probs @ mu_0
    return probs[0][0]


def p_obs_g_actions_initial(o_0, a_0):
    # Get real sensing action
    act_0 = prod_pomdp.actions[a_0]
    mu_0 = prod_pomdp.initial_dist
    # Obtain observable operators
    oo = observable_operator(o_0, act_0)
    # Creat a vector with all elements equals to 1
    one_vec = np.ones([1, prod_pomdp.state_size])
    # Initialize the probability of observation given sensing actions and initial states
    probs = one_vec @ oo
    probs = probs @ mu_0
    return probs[0][0]


def p_vtp1_obs_g_actions(v_tp1, y, a_list):
    # Get the real sensing actions
    act_list = [prod_pomdp.actions[a] for a in a_list]
    # Give value to the initial state
    mu_0 = prod_pomdp.initial_dist
    # Obtain observable operators
    oo = observable_operator(y[-1], act_list[-1])
    # Create a one-hot vecto which has a 1 element at state index v_{t+1}
    one_hot = np.zeros([1, prod_pomdp.state_size])
    one_hot[0][v_tp1] = 1
    # Initialize the probability of observation given sensing actions and initial states
    probs = one_hot @ oo
    # Calculate the probability
    for t in reversed(range(len(y) - 1)):
        oo = observable_operator(y[t], act_list[t])
        probs = probs @ oo
    # print(y)
    # print(senAct_list)
    # print(probs)
    probs = probs @ mu_0
    return probs[0][0]


def p_theta_obs(policy_net, y, a_list):
    """
    :param policy_net:
    :param y: the sequence of observations given states and actions
    :param a_list: the sequence of actions
    :return: the probability P(y, a_list ; theta)
    """
    m = fsc.memory_space.index('l')
    policy_prod = get_action_probability(policy_net, m, a_list[0])
    for i in range(len(y) - 1):
        o = fsc.observations.index(y[i])
        m = fsc.transition[m][o]
        policy_prod *= get_action_probability(policy_net, m, a_list[i + 1])
    p_obs_g_acts_initial = p_obs_g_actions_initial(y[0], a_list[0])
    p_obs_g_acts = p_obs_g_actions(y, a_list)
    return p_obs_g_acts / p_obs_g_acts_initial * policy_prod


def log_p_theta_obs(policy_net, y, a_list):
    """
    :param policy_net:
    :param y: the sequence of observations given states and sensing actions
    :param a_list: the sequence of sensing actions
    :return: the log probability log P(y, sa_list| s_0 ; theta)
    """
    m = fsc.memory_space.index('l')
    policy_sum = np.log2(get_action_probability(policy_net, m, a_list[0]))
    for i in range(len(y) - 1):
        o = fsc.observations.index(y[i])
        m = fsc.transition[m][o]
        policy_sum += np.log2(get_action_probability(policy_net, m, a_list[i + 1]))
    p_obs_g_acts_initial = p_obs_g_actions_initial(y[0], a_list[0])
    p_obs_g_acts = p_obs_g_actions(y, a_list)
    log_p_y_g_sas0 = np.log2(p_obs_g_acts) if p_obs_g_acts > 0 else float('-inf')
    # print(log_p_y_g_sas0)
    return log_p_y_g_sas0 - np.log2(p_obs_g_acts_initial) + policy_sum


def nabla_log_p_theta_obs(policy_net, y, a_list):
    m = fsc.memory_space.index('l')
    log_grad_sum = create_gradient_shaped_tensors(policy_net)
    log_grads = compute_log_policy_gradient(policy_net, m, a_list[0])
    for j in range(len(log_grad_sum)):
        log_grad_sum[j] = log_grads[j]
    for i in range(len(y) - 1):
        o = fsc.observations.index(y[i])
        m = fsc.transition[m][o]
        log_grads = compute_log_policy_gradient(policy_net, m, a_list[i + 1])
        for j in range(len(log_grad_sum)):
            log_grad_sum[j] += log_grads[j]
    return log_grad_sum


def p_zT_g_y(y, a_list):
    p_obs_g_acts = p_obs_g_actions(y, a_list)
    act_T = prod_pomdp.actions[a_list[-1]]
    o_T = y[-1]
    st_T = 'sink'
    v_T = prod_pomdp.states.index(st_T)
    emiss_prob = prod_pomdp.emiss[st_T][act_T][o_T]
    p_vtp1_obs_g_acts = p_vtp1_obs_g_actions(v_T, y[0:-1], a_list[0:-1])
    p_zT1 = emiss_prob * p_vtp1_obs_g_acts / p_obs_g_acts
    p_zT0 = 1 - p_zT1
    return p_zT1, p_zT0


def entropy_a_grad(policy_net, y_data, a_data):
    M = len(y_data)
    H = 0
    nabla_H = create_gradient_shaped_tensors(policy_net)
    for k in range(M):
        # start = time.time()
        y_k = y_data[k]
        a_list_k = a_data[k]
        # Get the values when z_T = 1
        # p_theta_yk = p_theta_obs(y_k, sa_list_k, theta)
        grad_log_P_theta_yk = nabla_log_p_theta_obs(policy_net, y_k, a_list_k)
        # print("gradient done")
        p_zT1, p_zT0 = p_zT_g_y(y_k, a_list_k)
        # print(p_zT1, p_zT0)

        # log2_p_theta_s0_yk = np.log2(p_theta_s0_yk) if p_theta_s0_yk > 0 else 0
        temp_H_1 = p_zT1 * np.log2(p_zT1) if p_zT1 > 0 else 0
        temp_H_0 = p_zT0 * np.log2(p_zT0) if p_zT0 > 0 else 0
        temp_H = temp_H_1 + temp_H_0
        H += temp_H
        # grad_p_theta_s0_yk = nabla_p_theta_s0_g_y(y_k, sa_list_k, s_0, theta)
        for j in range(len(nabla_H)):
            nabla_H[j] += temp_H * grad_log_P_theta_yk[j]
        # print("One iteration done")
        # end = time.time()
        # print(f"One trajectory done. It takes", end - start, "s")
    H = - H / M
    for j in range(len(nabla_H)):
        nabla_H[j] = - nabla_H[j] / M
    return H, nabla_H


def action_sampler(policy_net, m):
    prob_list = [get_action_probability(policy_net, m, a) for a in range(prod_pomdp.action_size)]
    return choices(prod_pomdp.actions, prob_list, k=1)[0]


def sample_data(policy_net, M, T):
    s_data = np.zeros([M, T], dtype=np.int32)
    a_data = []
    y_data = []
    for m in range(M):
        y = []
        a_list = []
        # start from initial state
        state = prod_pomdp.initial_states
        s = prod_pomdp.states.index(state)
        # Sample sensing action
        me = fsc.memory_space.index('l')
        act = action_sampler(policy_net, me)
        a = prod_pomdp.actions.index(act)
        a_list.append(a)
        # Get the observation of initial state
        obs = prod_pomdp.observation_function_sampler(state, act)
        o = fsc.observations.index(obs)
        me = fsc.transition[me][o]
        y.append(obs)

        for t in range(T):
            s = prod_pomdp.states.index(state)
            s_data[m, t] = s
            # sample the next state
            state = prod_pomdp.next_state_sampler(state, act)
            # Sample sensing action
            act = action_sampler(policy_net, me)
            a = prod_pomdp.actions.index(act)
            a_list.append(a)
            # Add the observation
            obs = prod_pomdp.observation_function_sampler(state, act)
            o = fsc.observations.index(obs)
            me = fsc.transition[me][o]
            y.append(obs)
        y_data.append(y)
        a_data.append(a_list)
    return s_data, y_data, a_data


def display_states_from_s_data(s_data):
    M = len(s_data)
    for k in range(M):
        s_list = s_data[k]
        print([prod_pomdp.states[s] for s in s_list])
    return 0


def main():
    # Define hyperparameters
    ex_num = 1
    iter_num = 2000  # iteration number of gradient ascent
    M = 200  # number of sampled trajectories
    T = 10  # length of a trajectory
    eta = 0.01  # step size for theta
    # kappa = 0.2  # step size for lambda
    # F = env.goals  # Define the goal region
    # alpha = 0.3  # value constraint
    state_dim = 1
    hidden_dim = 64

    policy_net = PolicyNetwork(state_dim, prod_pomdp.action_size, hidden_dim)
    # test_grads = create_gradient_shaped_tensors(policy_net)

    # Initialize the parameters
    # theta = np.random.random([fsc.memory_size, env.sensing_actions_size])
    # opt_values = value_iterations(1e-3, F)
    # theta = extract_opt_theta(opt_values, F)  # optimal theta initialization.
    # with open('backward_grid_world_1_data/Values/theta_3', 'rb') as f:
    #     theta = np.load(f, allow_pickle=True)

    # lam = np.random.uniform(1, 10)
    # Create empty lists
    entropy_list = []
    # theta_list = [theta]
    # Sample trajectories (observations)
    for i in range(iter_num):
        start = time.time()
        ##############################################
        s_data, y_data, a_data = sample_data(policy_net, M, T)
        # Gradient ascent process
        # print(y_data)
        # display_states_from_s_data(s_data)
        # SGD gradient
        approx_entropy, grad_H = entropy_a_grad(policy_net, y_data, a_data)
        # grad_H = torch.from_numpy(grad_H).type(dtype=torch.float32)
        # print("The gradient of entropy is", grad_H)
        print("The conditional entropy is", approx_entropy)
        entropy_list.append(approx_entropy)
        # SGD updates
        counter = 0
        for param in policy_net.parameters():
            if param.grad is not None:  # Ensure the gradient exists
                with torch.no_grad():
                    param += eta * grad_H[counter]  # Tensor of zeros
                counter += 1
        # theta = theta - eta * grad_H
        # theta_list.append(theta)
        ###############################################
        end = time.time()
        print(f"iteration_{i + 1} done. It takes", end - start, "s")
        print("#" * 100)

    # with open(f'./grid_world_2_data/Values/Correct_thetaList_{ex_num}', "wb") as pkl_wb_obj:
    #     pickle.dump(theta_list, pkl_wb_obj)

    with open(f'./deep_data/Values/entropy_{ex_num}', "wb") as pkl_wb_obj:
        pickle.dump(entropy_list, pkl_wb_obj)

    iteration_list = range(iter_num)
    plt.plot(iteration_list, entropy_list, label='entropy')
    plt.xlabel("The iteration number")
    plt.ylabel("entropy")
    plt.legend()
    plt.savefig(f'./deep_data/Graphs/Ex_{ex_num}.png')
    plt.show()


if __name__ == "__main__":
    main()
