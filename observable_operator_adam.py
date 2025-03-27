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

prod_pomdp = prod_pomdp()
fsc = FSC()


def pi_theta(theta, m, a):
    """
    :param m: the index of a finite sequence of observation, corresponding to K-step memory
    :param a: the sensing action to be given
    :param theta: the policy parameter, the memory_size * sensing_action_size
    :return: the Gibbs policy given the finite memory
    """
    e_x = np.exp(theta[m, :] - np.max(theta[m, :]))
    return (e_x / e_x.sum(axis=0))[a]


def log_policy_gradient(theta, m, a):
    # A memory space for K-step memory policy
    memory_space = fsc.memory_space
    memory_size = fsc.memory_size
    gradient = np.zeros([memory_size, prod_pomdp.action_size])
    memory = memory_space[m]
    senAct = prod_pomdp.actions[a]
    for m_prime in range(memory_size):
        for a_prime in range(prod_pomdp.action_size):
            memory_p = memory_space[m_prime]
            senAct_p = prod_pomdp.actions[a_prime]
            indicator_m = 0
            indicator_a = 0
            if memory == memory_p:
                indicator_m = 1
            if senAct == senAct_p:
                indicator_a = 1
            partial_pi_theta = indicator_m * (indicator_a - pi_theta(theta, m_prime, a_prime))
            gradient[m_prime, a_prime] = partial_pi_theta
    return gradient


def get_observable_operator():
    oo_dict = {}
    for obs_t in prod_pomdp.observations:
        oo_dict[obs_t] = {}
        for act_t in prod_pomdp.actions:
            oo = np.zeros([prod_pomdp.state_size, prod_pomdp.state_size])
            for i in range(prod_pomdp.state_size):
                for j in range(prod_pomdp.state_size):
                    st = prod_pomdp.states[i]
                    st_prime = prod_pomdp.states[j]
                    # The definition of observable operators
                    # print(st_prime, act_t)
                    if st in prod_pomdp.next_supp[st_prime][act_t]:
                        oo[i, j] = prod_pomdp.transition[st_prime][act_t][st] * prod_pomdp.emiss[st_prime][act_t][obs_t]
            oo_dict[obs_t][act_t] = oo
    return oo_dict


def p_obs_g_actions(y, a_list, observable_operator):
    # Get the real sensing actions
    act_list = [prod_pomdp.actions[a] for a in a_list]
    # Give value to the initial state
    mu_0 = prod_pomdp.initial_dist
    # Obtain observable operators
    oo = observable_operator[y[-1]][act_list[-1]]
    # Create a vector with all elements equals to 1
    one_vec = np.ones([1, prod_pomdp.state_size])
    # Initialize the probability of observation given sensing actions and initial states
    probs = one_vec @ oo
    # Calculate the probability
    for t in reversed(range(len(y) - 1)):
        oo = observable_operator[y[t]][act_list[t]]
        probs = probs @ oo
    # print(y)
    # print(senAct_list)
    # print(probs)
    probs = probs @ mu_0
    return probs[0][0]


def p_obs_g_actions_initial(o_0, a_0, observable_operator):
    # Get real sensing action
    act_0 = prod_pomdp.actions[a_0]
    mu_0 = prod_pomdp.initial_dist
    # Obtain observable operators
    oo = observable_operator[o_0][act_0]
    # Creat a vector with all elements equals to 1
    one_vec = np.ones([1, prod_pomdp.state_size])
    # Initialize the probability of observation given sensing actions and initial states
    probs = one_vec @ oo
    probs = probs @ mu_0
    return probs[0][0]


def p_vtp1_obs_g_actions(v_tp1, y, a_list, observable_operator):
    # Get the real sensing actions
    act_list = [prod_pomdp.actions[a] for a in a_list]
    # Give value to the initial state
    mu_0 = prod_pomdp.initial_dist
    # Obtain observable operators
    oo = observable_operator[y[-1]][act_list[-1]]
    # Create a one-hot vecto which has a 1 element at state index v_{t+1}
    one_hot = np.zeros([1, prod_pomdp.state_size])
    one_hot[0][v_tp1] = 1
    # Initialize the probability of observation given sensing actions and initial states
    probs = one_hot @ oo
    # Calculate the probability
    for t in reversed(range(len(y) - 1)):
        oo = observable_operator[y[t]][act_list[t]]
        probs = probs @ oo
    # print(y)
    # print(senAct_list)
    # print(probs)
    probs = probs @ mu_0
    return probs[0][0]


def p_theta_obs(theta, y, a_list, observable_operator):
    """
    :param theta: policy parameter
    :param y: the sequence of observations given states and actions
    :param a_list: the sequence of actions
    :return: the probability P(y, a_list ; theta)
    """
    m = fsc.memory_space.index('l')
    policy_prod = pi_theta(theta, m, a_list[0])
    for i in range(len(y) - 1):
        o = fsc.observations.index(y[i])
        m = fsc.transition[m][o]
        policy_prod *= pi_theta(theta, m, a_list[i + 1])
    p_obs_g_acts_initial = p_obs_g_actions_initial(y[0], a_list[0], observable_operator)
    p_obs_g_acts = p_obs_g_actions(y, a_list, observable_operator)
    return p_obs_g_acts / p_obs_g_acts_initial * policy_prod


def log_p_theta_obs(theta, y, a_list, observable_operator):
    """
    :param theta: policy parameter
    :param y: the sequence of observations given states and sensing actions
    :param a_list: the sequence of sensing actions
    :return: the log probability log P(y, sa_list| s_0 ; theta)
    """
    m = fsc.memory_space.index('l')
    policy_sum = np.log2(pi_theta(theta, m, a_list[0]))
    for i in range(len(y) - 1):
        o = fsc.observations.index(y[i])
        m = fsc.transition[m][o]
        policy_sum += np.log2(pi_theta(theta, m, a_list[i + 1]))
    p_obs_g_acts_initial = p_obs_g_actions_initial(y[0], a_list[0], observable_operator)
    p_obs_g_acts = p_obs_g_actions(y, a_list, observable_operator)
    log_p_y_g_sas0 = np.log2(p_obs_g_acts) if p_obs_g_acts > 0 else float('-inf')
    # print(log_p_y_g_sas0)
    return log_p_y_g_sas0 - np.log2(p_obs_g_acts_initial) + policy_sum


def nabla_log_p_theta_obs(theta, y, a_list):
    m = fsc.memory_space.index('l')
    log_grad_sum = log_policy_gradient(theta, m, a_list[0])
    for i in range(len(y) - 1):
        o = fsc.observations.index(y[i])
        m = fsc.transition[m][o]
        log_grad_sum += log_policy_gradient(theta, m, a_list[i + 1])
    return log_grad_sum


def p_zT_g_y(y, a_list, observable_operator):
    p_obs_g_acts = p_obs_g_actions(y, a_list, observable_operator)
    act_T = prod_pomdp.actions[a_list[-1]]
    o_T = y[-1]
    st_T = 'sink'
    v_T = prod_pomdp.states.index(st_T)
    emiss_prob = prod_pomdp.emiss[st_T][act_T][o_T]
    p_vtp1_obs_g_acts = p_vtp1_obs_g_actions(v_T, y[0:-1], a_list[0:-1], observable_operator)
    p_zT1 = emiss_prob * p_vtp1_obs_g_acts / p_obs_g_acts
    p_zT0 = 1 - p_zT1
    return p_zT1, p_zT0


def entropy_a_grad(theta, y_data, a_data, observable_operator):
    M = len(y_data)
    H = 0
    P = 0
    nabla_H = np.zeros([fsc.memory_size, prod_pomdp.action_size])
    nabla_P = np.zeros([fsc.memory_size, prod_pomdp.action_size])
    for k in range(M):
        # start = time.time()
        y_k = y_data[k]
        a_list_k = a_data[k]
        # Get the values when z_T = 1
        p_theta_yk = p_theta_obs(theta, y_k, a_list_k, observable_operator)
        grad_log_P_theta_yk = nabla_log_p_theta_obs(theta, y_k, a_list_k)
        # print("gradient done")
        p_zT1, p_zT0 = p_zT_g_y(y_k, a_list_k, observable_operator)
        # print(p_zT1, p_zT0)

        # log2_p_theta_s0_yk = np.log2(p_theta_s0_yk) if p_theta_s0_yk > 0 else 0
        temp_H_1 = p_zT1 * np.log2(p_zT1) if p_zT1 > 0 else 0
        temp_H_0 = p_zT0 * np.log2(p_zT0) if p_zT0 > 0 else 0
        temp_H = temp_H_1 + temp_H_0
        H += temp_H
        P += p_zT1
        # grad_p_theta_s0_yk = nabla_p_theta_s0_g_y(y_k, sa_list_k, s_0, theta)
        nabla_H += temp_H * grad_log_P_theta_yk
        nabla_P += p_zT1 * grad_log_P_theta_yk
        # print("One iteration done")
        # end = time.time()
        # print(f"One trajectory done. It takes", end - start, "s")
    H = - H / M
    P = P / M
    nabla_H = - nabla_H / M
    nabla_P = nabla_P / M
    return H, nabla_H, P, nabla_P


def entropy_a_grad_per_iter(theta, y_k, a_list_k, observable_operator):
    nabla_H = np.zeros([fsc.memory_size, prod_pomdp.action_size])
    nabla_P = np.zeros([fsc.memory_size, prod_pomdp.action_size])
    grad_log_P_theta_yk = nabla_log_p_theta_obs(theta, y_k, a_list_k)
    p_zT1, p_zT0 = p_zT_g_y(y_k, a_list_k, observable_operator)
    temp_H_1 = p_zT1 * np.log2(p_zT1) if p_zT1 > 0 else 0
    temp_H_0 = p_zT0 * np.log2(p_zT0) if p_zT0 > 0 else 0
    H = temp_H_1 + temp_H_0
    P = p_zT1
    nabla_H += H * grad_log_P_theta_yk
    nabla_P += p_zT1 * grad_log_P_theta_yk
    return H, nabla_H, P, nabla_P


def entropy_a_grad_multi(theta, y_data, a_data, observable_operator):
    M = len(y_data)
    H = 0
    P = 0
    nabla_H = np.zeros([fsc.memory_size, prod_pomdp.action_size])
    nabla_P = np.zeros([fsc.memory_size, prod_pomdp.action_size])
    with ProcessPoolExecutor(max_workers=24) as exe:
        H_a_gradH_list = exe.map(entropy_a_grad_per_iter, repeat(theta), y_data, a_data, repeat(observable_operator))
    for H_tuple in H_a_gradH_list:
        H += H_tuple[0]
        nabla_H += H_tuple[1]
        P += H_tuple[2]
        nabla_P += H_tuple[3]
    H = - H / M
    nabla_H = - nabla_H / M
    P = P / M
    nabla_P = nabla_P / M
    return H, nabla_H, P, nabla_P


def action_sampler(theta, m):
    prob_list = [pi_theta(theta, m, a) for a in range(prod_pomdp.action_size)]
    return choices(prod_pomdp.actions, prob_list, k=1)[0]


def sample_data(theta, M, T):
    s_data = np.zeros([M, T], dtype=np.int32)
    a_data = []
    y_data = []
    for m in range(M):
        y = []
        a_list = []
        # start from initial state
        state = prod_pomdp.initial_states
        state = choices(prod_pomdp.initial_states, prod_pomdp.initial_dist_sampling, k=1)[0]
        # Sample sensing action
        me = fsc.memory_space.index('l')
        act = action_sampler(theta, me)
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
            act = action_sampler(theta, me)
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
    ex_num = 4
    iter_num = 1000  # iteration number of gradient ascent
    M = 1000  # number of sampled trajectories
    T = 10  # length of a trajectory
    eta = 0.5  # step size for theta
    alpha = 1
    # state_dim = 1
    # hidden_dim = 64

    # policy_net = PolicyNetwork(state_dim, prod_pomdp.action_size, hidden_dim)
    # test_grads = create_gradient_shaped_tensors(policy_net)

    # Initialize the parameters
    theta = np.random.random([fsc.memory_size, prod_pomdp.action_size])
    # opt_values = value_iterations(1e-3, F)
    # theta = extract_opt_theta(opt_values, F)  # optimal theta initialization.
    # with open('backward_grid_world_1_data/Values/theta_3', 'rb') as f:
    #     theta = np.load(f, allow_pickle=True)
    obs_dict = get_observable_operator()
    # lam = np.random.uniform(1, 10)
    # Create empty lists
    entropy_list = []
    probs_list = []
    theta_list = [theta]
    # for adam update
    small_epsilon = 1e-8
    moment =0
    # Sample trajectories (observations)
    for i in range(iter_num):
        start = time.time()
        ##############################################
        s_data, y_data, a_data = sample_data(theta, M, T)
        # Gradient ascent process
        print(y_data)
        # display_states_from_s_data(s_data)
        # SGD gradient
        approx_entropy, grad_H, approx_P_Z1, grad_P = entropy_a_grad_multi(theta, y_data, a_data, obs_dict)
        grad = grad_H - alpha * grad_P
        # grad_H = torch.from_numpy(grad_H).type(dtype=torch.float32)
        # print("The gradient of entropy is", grad_H)
        print("The conditional entropy is", approx_entropy)
        entropy_list.append(approx_entropy)
        print("The probability of completing the task is", approx_P_Z1)
        probs_list.append(approx_P_Z1)
        # SGD updates
        moment +=  grad ** 2
        theta = theta - eta * grad/(np.sqrt(moment) + small_epsilon)
        theta_list.append(theta)
        ###############################################
        end = time.time()
        print(f"iteration_{i + 1} done. It takes", end - start, "s")
        print("#" * 100)

    with open(f'./data/Values/thetaList_{ex_num}', "wb") as pkl_wb_obj:
        pickle.dump(theta_list, pkl_wb_obj)

    with open(f'./data/Values/PZList_{ex_num}', "wb") as pkl_wb_obj:
        pickle.dump(probs_list, pkl_wb_obj)

    with open(f'./data/Values/entropy_{ex_num}', "wb") as pkl_wb_obj:
        pickle.dump(entropy_list, pkl_wb_obj)

    iteration_list = range(iter_num)
    plt.plot(iteration_list, entropy_list, label='entropy')
    plt.plot(iteration_list, probs_list, label=r'$P_\theta(Z_T = 1)$')
    plt.xlabel("The iteration number")
    plt.ylabel("Values")
    plt.legend()
    plt.savefig(f'./data/Graphs/Ex_{ex_num}.png')
    plt.show()

import os
if __name__ == "__main__":
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    main()

