from itertools import product
import numpy as np
import torch
WIDTH = 10
LENGTH = 10

class Environment:
    def __init__(self):
        # Define the states
        self.states_UAV = [(i , j) for i in range(WIDTH) for j in range(LENGTH)]
        self.states_Robot = [(i , j) for i in range(WIDTH) for j in range(LENGTH)]
        self.Types_Robot = [0, 1] # 0: normal; 1: adversarial
        # Define the cells
        self.trees = [(0, 2), (0, 3), (0, 8), (1, 3), (1, 9), (2, 0), (2, 1), 
                      (2, 2), (2, 3), (2, 8), (3, 0), (3, 1), (3, 3), (3, 4), 
                      (3, 8), (4, 0), (4, 1), (4, 2), (4, 3), (4, 9), (5, 2), (5, 9)]
        self.grasses = [(7, 5), (7, 6), (7, 7), (8, 2), (8, 3), (8, 5), (8, 6), (8, 7), 
                        (9, 0), (9, 1), (9, 2), (9, 3), (9, 4)]
        self.rocks = [(1, 6), (2, 6), (2, 7), (3, 7), (5, 7), (6, 7), (6, 8)]
        self.ponds = [(0, 4), (2, 4), (3, 2), (5, 5), (6, 1), (8, 4), (9, 7)]
        self.states_Robot = list(set(self.states_Robot) - set(self.rocks))
        
        # Define the sensors
        self.stationary_sensor_pos = [(5, 0), (4, 4), (8, 4)]
        self.stationary_sensor_range = [[(0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (5, 0), (6, 0), (7, 0), (8, 0), (9, 0)], 
                                        [(3, 3), (3, 4), (3, 5), (4, 3), (4, 4), (4, 5), (5, 3), (5, 4), (5, 5)], 
                                        [(7, 3), (7, 4), (7, 5), (8, 3), (8, 4), (8, 5), (9, 3), (9, 4), (9, 5)]]
        # Define observations
        self.observations = ['1', '2', '3', 'n']
        self.observations_size = len(self.observations)
        self.obs_noise_stationary = 0.1
        self.obs_noise_UAV = 0.1
        
        
        # Actions First index: down and up; Second index: right and left 
        self.actions_UAV = [(1, 0), (-1, 0), (0, 0), (0, 1), (0, -1)]           
        self.actions_Robots = [(1, 0), (-1, 0), (0, 0), (0, 1), (0, -1)]  
        
        # Goals
        self.normal_goal_Robot = [(9, 0)]
        self.adversarial_goal_Robot = [(2, 0)]
        self.goal_UAV = [(1, 2), (1, 7)]
        
        # Labeling function
        self.label_func = self.get_labeling_func()
        
        # Transition probability
        self.stochastic_prob = 0.2  # 20% chance move to one of two nearest cells (UAV)
        self.transition_UAV = self.get_transition(self.states_UAV, self.actions_UAV, self.stochastic_prob)  # UAV
        self.stochastic_prob = 0.1  # 10% chance move to one of two nearest cells (ground robot)
        self.transition_robot = self.get_transition(self.states_Robot, self.actions_Robots, self.stochastic_prob)  # Robot
        
        # Policy
        self.value_Robot_normal = self.value_iteration(0.01, self.normal_goal_Robot, self.ponds, self.states_Robot, self.actions_Robots, self.transition_robot, gamma = 0.8)
        self.policy_Robot_normal = self.get_policy(self.value_Robot_normal, self.normal_goal_Robot, self.ponds, self.states_Robot, self.actions_Robots, self.transition_robot, gamma=0.8)
        self.value_Robot_adversarial = self.value_iteration(0.01, self.adversarial_goal_Robot, self.ponds, self.states_Robot, self.actions_Robots, self.transition_robot, gamma = 0.8)
        self.policy_Robot_adversarial = self.get_policy(self.value_Robot_adversarial, self.adversarial_goal_Robot, self.ponds, self.states_Robot, self.actions_Robots, self.transition_robot, gamma=0.8)
        # self.value_UAV = 
        # self.policy_UAV = []
        
        # Policy-induced Markov chain
        self.robot_normal_P_M_C = self.ground_robot_P_M_C(self.transition_robot, self.policy_Robot_normal, self.states_Robot, self.actions_Robots)
        self.robot_adversarial_P_M_C = self.ground_robot_P_M_C(self.transition_robot, self.policy_Robot_adversarial, self.states_Robot, self.actions_Robots)
        
        # Get emission function
        self.Stationary_emission = self.get_emission_fun_stationary()
        self.UAV_emission = self.get_emission_fun_UAV()
        
        
        
#============================================================================================
    def get_transition(self, state_set, action_set, stochastic_prob):
        # trans[state][action][state]: first is current state, third is next state 
        sto_P = stochastic_prob
        trans = {}
        for state in state_set:
            trans[state] = {}
            for action in action_set:
                if action == (0, 0):     # Stationary 100% prob
                    trans[state][action] = {}
                    trans[state][action][state] = 1
                else:                   # Non-stationary consider 1. Valid state (if invalid, stay), 2 Edge bouncing, 3. stochastic dynamic 
                    trans[state][action] = {}
                    trans[state][action][state] = 0
                    temp_state = tuple(a + b for a, b in zip(state, action))
                    if temp_state in state_set:
                        trans[state][action][temp_state] = 1 - sto_P    # intended direction   80%   10% for one adj cell and 10% for another
                    else:
                        trans[state][action][state] += 1 - sto_P
                    if action[0] == 0:
                        no_int_action_set = [(1, 0), (-1, 0)]
                    else:
                        no_int_action_set = [(0, 1), (0, -1)]
                    for no_int_action in no_int_action_set:
                        temp_state_no_int = tuple(a + b for a, b in zip(state, no_int_action))
                        if temp_state_no_int in state_set:
                            trans[state][action][temp_state_no_int] = sto_P/2
                        else:
                            trans[state][action][state] += sto_P/2
        return trans
#============================================================================================        
        
    def reward(self, state, goals, penalty_state):
        if state in goals:
            return 20
        elif state in penalty_state:
            return -2
        else:
            return -0.2
    

    def value_iteration(self, threshold, goal, penalty_state_set, state_set, action_set, trans, gamma=0.8):
        # Select device: Use GPU if available, otherwise fallback to CPU
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize V(s) = 0 and move all data to GPU
        values = torch.zeros(len(state_set), device=device)
        values_pre = values.clone()

        # Compute rewards for all states and convert them into a tensor
        reward_tensor = torch.tensor(
            [self.reward(state, goal, penalty_state_set) for state in state_set], device=device
        )

        Delta = threshold + 0.1  # Set an initial value greater than the threshold
        while Delta > threshold:
            values_pre = values.clone()  # Store the previous iteration's values
            new_values = torch.zeros_like(values)  # Initialize new values

            # Iterate through all states and compute value function updates
            for state in state_set:
                V_n = float('-inf')  # Initialize as negative infinity
                for action in action_set:
                    P_s_a_s_prime = trans[state][action]  # Get transition probabilities P(s' | s, a)
                    temp_V = 0
                    for state_prime in P_s_a_s_prime.keys():  # Iterate over all possible next states s'
                        state_prime_index = state_set.index(state_prime)
                        temp_V += P_s_a_s_prime[state_prime] * (reward_tensor[state_set.index(state)] + gamma * values_pre[state_prime_index])
                    if temp_V > V_n:
                        V_n = temp_V
                new_values[state_set.index(state)] = V_n  # Update the value for this state

            values = new_values
            Delta = torch.max(torch.abs(values - values_pre))  # Compute convergence condition

        return values.cpu().numpy()  # Convert back to NumPy for further processing

    
    
    def get_policy(self, opt_value, goal, penalty_state_set, state_set, action_set, trans, gamma=0.8):
        policy = state_set.copy()          # Pi(s) = optimal action, greedy policy
        # print(policy)
        for state in state_set:
            summation_value = np.zeros(len(action_set))
            i = 0
            for action in action_set:
                state_Prime_dist = trans[state][action]
                for state_Prime in state_Prime_dist.keys():
                    state_prime_prob = state_Prime_dist[state_Prime]
                    summation_value[i] += state_prime_prob*(self.reward(state, goal, penalty_state_set) + gamma*opt_value[state_set.index(state_Prime)])
                i = i + 1
            policy[state_set.index(state)] = action_set[np.argmax(summation_value)]
        return policy
    
    
    def ground_robot_P_M_C(self, trans, policy, state_set, action_set):    # MDP == Dynamic 
        P_M_C = {s: {s_prime: 0 for s_prime in state_set} for s in state_set}
        for state in state_set:  
            if state not in trans:  
                # print(f"Warning: state {state} not found in trans.")
                continue

            state_idx = state_set.index(state)
            action = policy[state_idx]  

            if action not in trans[state]:  
                # print(f"Warning: action {action} not found for state {state} in trans.")
                continue

            for s_prime in state_set:  
                if s_prime in trans[state][action]:  
                    P_M_C[state][s_prime] = trans[state][action][s_prime]
                # else:
                #     print(f"Warning: next_state {s_prime} not found for (state {state}, action {action}).")
        return P_M_C
    
#============================================================================================  

    def observation_function_stationary(self, state, sAct):
        if state in self.stationary_sensor_range[0] and self.observations[0] == sAct:
            return [self.observations[0], self.observations[3]]
        elif state in self.stationary_sensor_range[1] and self.observations[1] == sAct:
            return [self.observations[1], self.observations[3]]
        elif state in self.stationary_sensor_range[2] and self.observations[2] == sAct:
            return [self.observations[2], self.observations[3]]
        else:
            return [self.observations[3]]
        
    def emission_function_stationary(self, state, sAct, o):      # E(o | s, a)
        observation_set = self.observation_function_stationary(state, sAct)
        if o in observation_set:
            if o == self.observations[3] and len(observation_set) == 1:
                return 1
            elif o == self.observations[3] and len(observation_set) == 2:
                return self.obs_noise_stationary
            else:
                return 1 - self.obs_noise_stationary
        else:
            return 0
        
    def get_emission_fun_stationary(self):
        emission_func = {}
        for state in self.states_Robot:
            for sAct in self.observations:
                for o in self.observations:
                    prob = self.emission_function_stationary(state, sAct, o)
                    emission_func[(state, sAct, o)] = prob
        return emission_func
        
    def observation_function_UAV(self, gr_state, UAV_state): # 返回絕對位置, 不是相對位置
        x, y = UAV_state
        gx, gy = gr_state
        
        L_infinity_norm = np.max([abs(gx - x), abs(gy - y)])
        # print(L_infinity_norm)

        if L_infinity_norm <=1:
            return gr_state       
        else:
            return None
            
    def emission_function_UAV(self, gr_state, UAV_state, o):            # E(o | s)   UAV does not have action to select the sensors
        observation_pos = self.observation_function_UAV(gr_state, UAV_state)
        # Set probability 1 for the observed position
        if observation_pos is o:
            return 1 - self.obs_noise_UAV  # 0.9 對上 0.1 如果沒對上  
        else:
            return self.obs_noise_UAV
        
    def get_emission_fun_UAV(self):
        emission_func = {}
        for gr_state in self.states_Robot:
            for UAV_state in self.states_UAV:
                for o in self.states_Robot:
                    prob = self.emission_function_UAV(gr_state, UAV_state, o)
                    emission_func[(gr_state, UAV_state, o)] = prob
        return emission_func
        
    # G: robot at normal goal; Z: robot at adversarial goal; S: UAV see drone
    # A: UAV at flag A; B UAV at flag B
    def get_labeling_func(self):
        label_func = {}
        for gr_state in self.states_Robot:
            for UAV_state in self.states_UAV:
                for tau in self.Types_Robot:
                    key = (gr_state, UAV_state, tau)
                    label_func[key] = set()

                    if tau == 0 and gr_state in self.normal_goal_Robot:
                        label_func[key].add('G')
                    if tau == 1 and gr_state in self.adversarial_goal_Robot:
                        label_func[key].add('Z')

                    gr_r, gr_c = gr_state
                    uav_r, uav_c = UAV_state
                    if abs(gr_r - uav_r) <= 1 and abs(gr_c - uav_c) <= 1:
                        label_func[key].add('S')

                    if UAV_state == self.goal_UAV[0]:
                        label_func[key].add('B')

                    if UAV_state == self.goal_UAV[1]:
                        label_func[key].add('A')

        return label_func
        
        
def main():
    env = Environment()

    # Check transition should be 1
    # trans = env.transition_UAV
    # incorrect_num = 0
    # for state in trans.keys():
    #     for action in trans[state].keys():
    #         print("State: ", state, "Action: ", action, "Prob: ", sum(trans[state][action].values()))
    #         if sum(trans[state][action].values()) !=1:
    #             print("This transition is incorrect!")
    #             incorrect_num += 1
    # print("# of incorrect trans: ", incorrect_num)

    # Check Value iteration and Policies
    # print("Normal Robot")
    # for state, value, policy in zip(env.states_Robot, env.value_Robot_normal, env.policy_Robot_normal):
    #     print("State: ", state, "Value", value, "Policy", policy)
    # print("------------------------------------------------------------------------------------------")       
    # print("Adversarial Robot")
    # for state, value, policy in zip(env.states_Robot, env.value_Robot_adversarial, env.policy_Robot_adversarial):
    #     print("State: ", state, "Value", value, "Policy", policy)        


    # Check Policy-induced Markov chain
    # current_state = (2, 1)
    # next_state = (2, 0)
    # print(env.robot_normal_P_M_C[current_state][next_state])
    # print(env.robot_adversarial_P_M_C[current_state][next_state])


    # Check observation and emission functions
    UAV_state = (3, 3)
    Robot_state = (3, 2)
    observation = (3, 2)
    join_state = (Robot_state, UAV_state, observation)
    print(env.UAV_emission[join_state])


    # Check labeling function
    # Robot_state = (9, 0)
    # UAV_state = (1, 6)
    # tau = 0
    # join_state = (Robot_state, UAV_state, tau)
    # print(env.label_func[join_state])

            
if __name__ == "__main__":
    main()