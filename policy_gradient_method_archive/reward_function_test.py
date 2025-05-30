from policy_gradient_method_archive.observable_operator import *


def p_sT_g_y(st_T, y, a_list, observable_operator):
    p_obs_g_acts = p_obs_g_actions(y, a_list, observable_operator)
    act_T = prod_pomdp.actions[a_list[-1]]
    o_T = y[-1]
    v_T = prod_pomdp.states.index(st_T)
    emiss_prob = prod_pomdp.emiss[st_T][act_T][o_T]
    if len(y) > 1:
        p_vtp1_obs_g_acts = p_vtp1_obs_g_actions(v_T, y[0:-1], a_list[0:-1], observable_operator)
    else:
        p_vtp1_obs_g_acts = prod_pomdp.initial_dist[v_T]
    p_sT1 = emiss_prob * p_vtp1_obs_g_acts / p_obs_g_acts
    return p_sT1


def policy_theta(theta, s, a):
    """
    :param m: the index of a finite sequence of observation, corresponding to K-step memory
    :param a: the sensing action to be given
    :param theta: the policy parameter, the memory_size * sensing_action_size
    :return: the Gibbs policy given the finite memory
    """
    e_x = np.exp(theta[s, :] - np.max(theta[s, :]))
    return (e_x / e_x.sum(axis=0))[a]


def get_transition_matrix(theta, T):
    tm_dict = {}
    for t in range(T):
        tm = np.zeros([prod_pomdp.state_size, prod_pomdp.state_size])
        tm_prod = np.ones([prod_pomdp.state_size, prod_pomdp.state_size])
        for i in range(prod_pomdp.state_size):
            for j in range(prod_pomdp.state_size):
                st = prod_pomdp.states[i]
                st_prime = prod_pomdp.states[j]
                for act_t in prod_pomdp.actions:
                    a = prod_pomdp.actions.index(act_t)
                    if st_prime in prod_pomdp.next_supp[st][act_t]:
                        tm[i, j] = prod_pomdp.transition[st][act_t][st_prime] * policy_theta(theta, i, a)
        tm_prod = tm_prod * tm
        tm_dict[t] = tm_prod
    return tm_dict


def p_z_g_st(st_t, t, tm_dict):
    tm = tm_dict[t]
    prob = 0
    for st_T in prod_pomdp.secret_states:
        v_T_prime = prod_pomdp.states.index(st_T)
        v_t = prod_pomdp.states.index(st_t)
        prob += tm[v_t][v_T_prime]
    return prob


def p_z_y_Y(t, y, a_list, observable_operator, tm_dict):
    prob = 0
    for st_t in prod_pomdp.states:
        p_s_g_y = p_sT_g_y(st_t, y, a_list, observable_operator)
        p_z_g_s = p_z_g_st(st_t, t, tm_dict)
        prob += p_z_g_s * p_s_g_y
    return prob


def entropy_like_reward_first(t, y, a_list, observable_operator, tm_dict):
    p2 = p_z_y_Y(t, y, a_list, observable_operator, tm_dict)
    entropy_2 = p2 * np.log2(p2) if p2 > 0 else 0
    return - entropy_2


def entropy_like_reward(t, y, a_list, observable_operator, tm_dict):
    y_c = y[0:-1]
    a_list_c = a_list[0:-1]
    p1 = p_z_y_Y(t, y_c, a_list_c, observable_operator, tm_dict)
    p2 = p_z_y_Y(t, y, a_list, observable_operator, tm_dict)
    entropy_1 = p1 * np.log2(p1) if p1 > 0 else 0
    entropy_2 = p2 * np.log2(p2) if p2 > 0 else 0
    return entropy_1 - entropy_2


def main():
    M = 1  # number of sampled trajectories
    T = 10  # length of a trajectory

    # Initialize the parameters
    reward_list = []
    theta_r = np.random.random([prod_pomdp.state_size, prod_pomdp.action_size])
    theta = np.random.random([fsc.memory_size, prod_pomdp.action_size - 1])
    obs_dict = get_observable_operator()
    tm_dict = get_transition_matrix(theta_r, T)
    s_data, y_data, a_data = sample_data(theta, M, T)
    y_list = y_data[0]
    a_list = a_data[0]
    cumu_reward = 0
    reward_list.append(0)
    reward = entropy_like_reward_first(1, y_list[0:1], a_list[0:1], obs_dict, tm_dict)
    cumu_reward += reward
    reward_list.append(reward)
    for t in range(2, T):
        reward = entropy_like_reward(t, y_list[0:t], a_list[0:t], obs_dict, tm_dict)
        cumu_reward += reward
        reward_list.append(reward)
    print(cumu_reward)
    entropy = entropy_a_grad(theta, y_data, a_data, obs_dict)[0]
    print(entropy)
    plt.bar(range(1, T), reward_list[1:], label=r'reward $R(y,\pi)$')
    plt.xlabel("The time t")
    plt.ylabel("Rewards values")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'./data_2/Graphs/rewards.png')
    plt.show()


if __name__ == "__main__":
    main()
