import numpy as np
from finite_state_controller import FSC
from pomdp import pomdp
from DFA import DFA
from product_pomdp import prod_pomdp
from observable_operator import sample_data
from observable_operator import display_states_from_s_data
import pickle

pomdp = pomdp()
dfa = DFA()
prod_pomdp = prod_pomdp()
fsc = FSC()

prod_pomdp.check_the_transition()

# theta = np.random.random([fsc.memory_size, prod_pomdp.action_size])
with open(f'data/Values/thetaList_1', "rb") as pkl_rb_obj:
    theta_list = pickle.load(pkl_rb_obj)
theta = theta_list[-1]

M = 100  # number of sampled trajectories
T = 10  # length of a trajectory
s_data, y_data, a_data = sample_data(theta, M, T)
display_states_from_s_data(s_data)
# print(a_data)
# print(y_data)

