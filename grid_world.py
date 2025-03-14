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
        self.Join_states = [(robot_x, robot_y, uav_x, uav_y, robot_type)
                            for (robot_x, robot_y), (uav_x, uav_y), robot_type
                            in product(self.states_Robot, self.states_UAV, self.Types_Robot)]
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
        
        # Actions First index: down and up; Second index: right and left 
        self.actions_UAV = [(1, 0), (-1, 0), (0, 0), (0, 1), (0, -1)]           
        self.actions_Robots = [(1, 0), (-1, 0), (0, 0), (0, 1), (0, -1)]  
        
        # Goals
        self.normal_goal_Robot = [(9, 0)]
        self.adversarial_goal_Robot = [(2, 0)]
        self.goal_UAV = [(1, 2), (1, 7)]
        
        # Transition probability
        self.stochastic_prob = 0.2  # 20% chance move to one of two nearest cells (UAV)
        self.transition_UAV = self.transition_construction(self.states_UAV, self.actions_UAV, self.stochastic_prob)  # UAV
        self.stochastic_prob = 0.1  # 10% chance move to one of two nearest cells (ground robot)
        self.transition_robot = self.transition_construction(self.states_Robot, self.actions_Robots, self.stochastic_prob)  # Robot
        
        # Policy
        self.value_Robot_normal = self.value_iteration(0.01, self.normal_goal_Robot, self.ponds, self.states_Robot, self.actions_Robots, self.transition_robot, gamma = 0.8)
        self.policy_Robot_normal = self.policy_extraction(self.value_Robot_normal, self.normal_goal_Robot, self.ponds, self.states_Robot, self.actions_Robots, self.transition_robot, gamma=0.8)
        self.value_Robot_adversarial = self.value_iteration(0.01, self.adversarial_goal_Robot, self.ponds, self.states_Robot, self.actions_Robots, self.transition_robot, gamma = 0.8)
        self.policy_Robot_adversarial = self.policy_extraction(self.value_Robot_adversarial, self.adversarial_goal_Robot, self.ponds, self.states_Robot, self.actions_Robots, self.transition_robot, gamma=0.8)
        # self.value_UAV = 
        # self.policy_UAV = []
        
        # Policy-induced Markov chain
        self.robot_normal_P_M_C = self.ground_robot_P_M_C(self.transition_robot, self.policy_Robot_normal, self.states_Robot, self.actions_Robots)
        self.robot_adversarial_P_M_C = self.ground_robot_P_M_C(self.transition_robot, self.policy_Robot_adversarial, self.states_Robot, self.actions_Robots)
        
        
        
#============================================================================================
    def transition_construction(self, state_set, action_set, stochastic_prob):
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

    
    
    # def value_iteration(self, threshold, goal, penalty_state_set, state_set, action_set, trans, gamma = 0.8):
    #     values = np.zeros(len(state_set))
    #     values_pre = np.copy(values)
    #     Delta = threshold + 0.1
    #     while Delta > threshold:
    #         for state in state_set:
    #             V_n = 0 # initial
    #             for action in action_set:
    #                 P_s_a_s_prime = trans[state][action]
    #                 temp_V = 0
    #                 for state_prime in P_s_a_s_prime.keys(): # every next state
    #                     state_prime_index = state_set.index(state_prime)
    #                     temp_V += P_s_a_s_prime[state_prime]*(self.reward(state, goal, penalty_state_set) + gamma*values[state_prime_index])
    #                 if temp_V > V_n:
    #                     V_n = temp_V
    #             state_index = state_set.index(state)
    #             values[state_index] = V_n
    #         Delta = np.max(values - values_pre)  # When the max value in the state vector is lower than threshold, we terminate it
    #         value_pre = np.copy(values)
    #     return values
    
    
    def policy_extraction(self, opt_value, goal, penalty_state_set, state_set, action_set, trans, gamma=0.8):
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

            
            
if __name__ == "__main__":
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
    current_state = (2, 1)
    next_state = (2, 0)
    print(env.robot_normal_P_M_C[current_state][next_state])
    print(env.robot_adversarial_P_M_C[current_state][next_state])
    