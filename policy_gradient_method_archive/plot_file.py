import matplotlib.pyplot as plt
import pickle

with open(f'../data_2/Values/thetaList_2', "rb") as pkl_rb_obj:
    theta_list = pickle.load(pkl_rb_obj)
# theta = theta_list[-1]

with open(f'../data_2/Values/PZList_2', "rb") as pkl_rb_obj:
    probs_list = pickle.load(pkl_rb_obj)
# pz = PZlist[-1]

with open(f'../data_2/Values/entropy_2', "rb") as pkl_rb_obj:
    entropy_list = pickle.load(pkl_rb_obj)
# entropy = entropy_list[-1]

iter_num = 1000
iteration_list = range(iter_num)
plt.plot(iteration_list, entropy_list, label=r'entropy $H(Z_T|Y;\theta)$')
plt.plot(iteration_list, probs_list, label=r'probability $P_\theta(W_T = 1)$')
plt.xlabel("The iteration number")
plt.ylabel("Values")
plt.legend()
plt.tight_layout()
plt.savefig(f'./data_2/Graphs/Ex_final_2.png')
plt.show()