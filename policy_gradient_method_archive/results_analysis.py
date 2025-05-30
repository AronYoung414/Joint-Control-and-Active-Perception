from policy_gradient_method_archive.observable_operator import *


def approx_P_ZT_g_y(y_data, a_data, observable_operator):
    M = len(y_data)
    P1 = 0
    P2 = 0
    P3 = 0
    for k in range(M):
        # start = time.time()
        y_k = y_data[k]
        a_list_k = a_data[k]
        p_zT1, p_zT0, p_wT1 = p_zT_g_y(y_k, a_list_k, observable_operator)
        P1 += p_zT1
        P2 += p_zT0
        P3 += p_wT1
    P1 = P1 / M
    P2 = P2 / M
    P3 = P3 / M
    return P1, P2, P3


# Define hyperparameters
iter_num = 1000  # iteration number of gradient ascent
M = 1000  # number of sampled trajectories
T = 10  # length of a trajectory

with open(f'../data_2/Values/thetaList_2', "rb") as pkl_rb_obj:
    theta_list = pickle.load(pkl_rb_obj)
theta = theta_list[-1]

with open(f'../data_2/Values/PZList_2', "rb") as pkl_rb_obj:
    PZlist = pickle.load(pkl_rb_obj)
pz = PZlist[-1]

with open(f'../data_2/Values/entropy_2', "rb") as pkl_rb_obj:
    entropy_list = pickle.load(pkl_rb_obj)
entropy = entropy_list[-1]

print("The probability of completing task", pz)
print("The entropy of secret", entropy)

s_data, y_data, a_data = sample_data(theta, M, T)
obs_dict = get_observable_operator()
p_zT1, p_zT0, p_WT1 = approx_P_ZT_g_y(y_data, a_data, obs_dict)
print("Posterior of our method", p_zT1, p_zT0)
print("The probability of completing task", p_WT1)

# with open(f'data/Values/thetaList_1', "rb") as pkl_rb_obj:
#     theta_list = pickle.load(pkl_rb_obj)
# theta = theta_list[-1]
#
# with open(f'data/Values/PZList_1', "rb") as pkl_rb_obj:
#     PZlist = pickle.load(pkl_rb_obj)
# pz = PZlist[-1]
#
# # print(pz)
#
# s_data, y_data, a_data = sample_data(theta, M, T)
# obs_dict = get_observable_operator()
# p_zT1, p_zT0 = approx_P_ZT_g_y(y_data, a_data, obs_dict)
# print("Posterior of our method", p_zT1, p_zT0)