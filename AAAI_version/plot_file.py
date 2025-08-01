import matplotlib.pyplot as plt
import pickle

# with open(f'../data_2/Values/thetaList_2', "rb") as pkl_rb_obj:
#     theta_list = pickle.load(pkl_rb_obj)
# theta = theta_list[-1]
ex_num = 5

with open(f'data/Values/PZList_{ex_num}', "rb") as pkl_rb_obj:
    probs_list = pickle.load(pkl_rb_obj)
pzs = probs_list[-50:]

with open(f'data/Values/entropy_{ex_num}', "rb") as pkl_rb_obj:
    entropy_list = pickle.load(pkl_rb_obj)
entropys = entropy_list[-50:]

pz = sum(pzs)/len(pzs)
entropy = sum(entropys)/len(entropys)

print("The last entropy is", entropy)
print("The last probability is", pz)

iter_num = 1000
iteration_list = range(iter_num)
plt.figure(figsize=(10, 8))
plt.rcParams.update({'font.size': 30})
plt.plot(iteration_list, entropy_list, label=r'entropy $H(Z_T|Y;\theta)$')
plt.plot(iteration_list, probs_list, label=r'probability $P_\theta(W_T = 1)$')
plt.xlabel("The iteration number")
plt.ylabel("Values")
plt.legend()
plt.tight_layout()
plt.savefig(f'./data/Graphs/result.pdf')
plt.show()