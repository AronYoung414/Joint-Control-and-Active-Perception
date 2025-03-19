from random import choices
import numpy as np


class graph:

    def __init__(self):
        # Define states
        self.gr_states = ['0', '1', '2', '3', '4', '5']
        self.uav_states = ['0', '1', '2', '3', '4', '5']
        # self.state_indices = list(range(len(self.states)))
        # self.state_size = len(self.states)
        # Define initial state
        self.gr_initial_state = '0'
        # self.initial_state_idx = self.states.index(self.initial_state)
        # Define actions
        self.gr_actions = ['a', 'b']
        self.uav_actions = ['a', 'b']
        # self.action_size = len(self.actions)
        # self.action_indices = list(range(len(self.actions)))
        # transition probability dictionary
        self.gr_transition = self.get_transition()
        self.gr_next_supp = self.get_next_supp()
        # Goals
        self.goals_n = ['3']  # The goal of nominal agent
        self.goals_a = ['4']  # The goal of adversary
        self.goal_reward = 1
        # reward dictionary
        self.rewards_n = self.get_rewards(self.goals_n)  # The rewards of nominal agent
        self.rewards_a = self.get_rewards(self.goals_a)  # The rewards of adversary
        # discounting factor
        self.gamma = 0.9
        self.opt_policy_n = self.value_iteration(self.rewards_n)[1]  # The optimal policy of nominal agent
        self.opt_policy_a = self.value_iteration(self.rewards_a)[1]  # The optimal policy of adversary
        # calculate the transition of ground robots under the optimal policy
        self.transition_n = self.get_states_transition('n')
        self.transition_a = self.get_states_transition('a')
        # Define UAV with sensors
        self.uav_initial_state = '5'
        self.uav_transition = self.get_transition()
        self.uav_next_supp = self.get_next_supp()
        self.uav_goal = '0'
        # self.sensors = [['1'], ['2', 'f_0'], ['f_1']]
        # Secrets
        # self.secrets = ['f_0', 'f_1']
        # self.secret_indices = [self.states.index(secret) for secret in self.secrets]

    @staticmethod
    def get_transition():
        trans = {'0': {'a': {'1': 0.9, '2': 0.1}, 'b': {'1': 0.1, '2': 0.9}},
                 '1': {'a': {'3': 0.9, '4': 0.1}, 'b': {'3': 0.1, '4': 0.9}},
                 '2': {'a': {'3': 0.9, '4': 0.1}, 'b': {'3': 0.1, '4': 0.9}},
                 '3': {'a': {'5': 1}, 'b': {'5': 1}},
                 '4': {'a': {'5': 1}, 'b': {'5': 1}},
                 '5': {'a': {'3': 0.9, '4': 0.1}, 'b': {'3': 0.1, '4': 0.9}}}
        return trans

    def get_next_supp(self):
        next_supp = {}
        for st in self.gr_states:
            next_supp[st] = set()
            for act in self.gr_actions:
                next_supp[st] = next_supp[st].union(set(self.gr_transition[st][act].keys()))
        return next_supp

    def get_rewards(self, goals):
        rewards = {}
        for st in self.gr_states:
            rewards[st] = {}
            for act in self.gr_actions:
                if st in goals:
                    rewards[st][act] = self.goal_reward
                else:
                    rewards[st][act] = 0
        return rewards

    def extract_policy(self, rewards, V):
        """
        Extracts the optimal policy given the optimal value function.

        :param states: List of states
        :param actions: List of actions
        :param transition_probabilities: Dictionary with keys (state, action, next_state) and values as probabilities
        :param rewards: Dictionary with keys (state, action, next_state) and values as rewards
        :param V: Optimal value function
        :param gamma: Discount factor
        :return: Optimal deterministic policy as a dictionary {state: best_action}
        """
        policy = {}
        for st in self.gr_states:
            best_action = max(self.gr_actions, key=lambda act:
            rewards[st][act] + self.gamma * sum(
                self.gr_transition[st][act][st_prime] * V[st_prime]
                for st_prime in self.gr_transition[st][act].keys()
            ))
            policy[st] = {}
            for act in self.gr_actions:
                if act == best_action:
                    policy[st][act] = 1
                else:
                    policy[st][act] = 0
        return policy

    def value_iteration(self, rewards, theta=1e-6):
        """
        Performs Value Iteration to compute the optimal state values.

        :param states: List of states
        :param actions: List of actions
        :param transition_probabilities: Dictionary with keys (state, action, next_state) and values as probabilities
        :param rewards: Dictionary with keys (state, action) and values as rewards
        :param gamma: Discount factor
        :param theta: Convergence threshold
        :return: Optimal value function and optimal policy
        """
        V = {st: 0 for st in self.gr_states}  # Initialize value function to zero

        while True:
            delta = 0
            for st in self.gr_states:
                v = V[st]
                V[st] = max(
                    rewards[st][act] + self.gamma * sum(
                        self.gr_transition[st][act][st_prime] * V[st_prime]
                        for st_prime in self.gr_transition[st][act].keys())
                    for act in self.gr_actions
                )
                delta = max(delta, abs(v - V[st]))
            if delta < theta:
                break

        return V, self.extract_policy(rewards, V)

    def get_states_transition(self, type):
        trans = {}
        for st in self.gr_states:
            trans[st] = {}
            for st_prime in self.gr_next_supp[st]:
                trans[st][st_prime] = 0
                for act in self.gr_actions:
                    if type == 'n':
                        trans[st][st_prime] += self.gr_transition[st][act][st_prime] * self.opt_policy_n[st][act]
                    elif type == 'a':
                        trans[st][st_prime] += self.gr_transition[st][act][st_prime] * self.opt_policy_a[st][act]
                    else:
                        raise ValueError('Invalid type parameter.')
        return trans
