from observable_operator import *


def grad(theta, y_data, a_data, observable_operator):
    M = len(y_data)
    P = 0
    nabla_P = np.zeros([fsc.memory_size, prod_pomdp.action_size])
    for k in range(M):
        # start = time.time()
        y_k = y_data[k]
        a_list_k = a_data[k]
        # Get the values when z_T = 1
        grad_log_P_theta_yk = nabla_log_p_theta_obs(theta, y_k, a_list_k)
        # print("gradient done")
        p_zT1, p_zT0 = p_zT_g_y(y_k, a_list_k, observable_operator)
        # print(p_zT1, p_zT0)

        # log2_p_theta_s0_yk = np.log2(p_theta_s0_yk) if p_theta_s0_yk > 0 else 0
        P += p_zT1
        # grad_p_theta_s0_yk = nabla_p_theta_s0_g_y(y_k, sa_list_k, s_0, theta)
        nabla_P += p_zT1 * grad_log_P_theta_yk
        # print("One iteration done")
        # end = time.time()
        # print(f"One trajectory done. It takes", end - start, "s")
    P = P / M
    nabla_P = nabla_P / M
    return P, nabla_P


def grad_per_iter(theta, y_k, a_list_k, observable_operator):
    nabla_P = np.zeros([fsc.memory_size, prod_pomdp.action_size])
    grad_log_P_theta_yk = nabla_log_p_theta_obs(theta, y_k, a_list_k)
    p_zT1, p_zT0 = p_zT_g_y(y_k, a_list_k, observable_operator)
    P = p_zT1
    nabla_P += p_zT1 * grad_log_P_theta_yk
    return P, nabla_P


def grad_multi(theta, y_data, a_data, observable_operator):
    M = len(y_data)
    P = 0
    nabla_P = np.zeros([fsc.memory_size, prod_pomdp.action_size])
    with ProcessPoolExecutor(max_workers=24) as exe:
        H_a_gradH_list = exe.map(entropy_a_grad_per_iter, repeat(theta), y_data, a_data, repeat(observable_operator))
    for H_tuple in H_a_gradH_list:
        P += H_tuple[0]
        nabla_P += H_tuple[1]
    P = P / M
    nabla_P = nabla_P / M
    return P, nabla_P


def main():
    # Define hyperparameters
    ex_num = 1
    iter_num = 1000  # iteration number of gradient ascent
    M = 1000  # number of sampled trajectories
    T = 10  # length of a trajectory
    eta = 0.5  # step size for theta
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
    probs_list = []
    theta_list = [theta]
    # Sample trajectories (observations)
    for i in range(iter_num):
        start = time.time()
        ##############################################
        s_data, y_data, a_data = sample_data(theta, M, T)
        # Gradient ascent process
        print(y_data)
        # display_states_from_s_data(s_data)
        # SGD gradient
        approx_P_Z1, grad_P = grad(theta, y_data, a_data, obs_dict)
        # grad_H = torch.from_numpy(grad_H).type(dtype=torch.float32)
        # print("The gradient of entropy is", grad_H)
        print("The probability of completing the task is", approx_P_Z1)
        probs_list.append(approx_P_Z1)
        # SGD updates
        theta = theta + eta * grad_P
        theta_list.append(theta)
        ###############################################
        end = time.time()
        print(f"iteration_{i + 1} done. It takes", end - start, "s")
        print("#" * 100)

    with open(f'./baseline/Values/thetaList_{ex_num}', "wb") as pkl_wb_obj:
        pickle.dump(theta_list, pkl_wb_obj)

    with open(f'./baseline/Values/PZList_{ex_num}', "wb") as pkl_wb_obj:
        pickle.dump(probs_list, pkl_wb_obj)

    iteration_list = range(iter_num)
    plt.plot(iteration_list, probs_list, label=r'$P_\theta(Z_T = 1)$')
    plt.xlabel("The iteration number")
    plt.ylabel("Values")
    plt.legend()
    plt.savefig(f'./baseline/Graphs/Ex_{ex_num}.png')
    plt.show()


if __name__ == "__main__":
    main()
