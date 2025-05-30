import numpy as np
from product_pomdp import prod_pomdp
# from observation_prefix_tree_for_graph import SimpleObservationSequenceEnumerator
# from random import choices
# from random import choice
from itertools import permutations, product
# import pickle


class InformationRewards:

    def __init__(self):
        self.prod_pomdp = prod_pomdp()
        self.trans = self.get_trans()
        self.observable_operators = self.get_observable_operator()
        print("observable operators done.")
        # self.all_possible_sequences = self.all_sequences_up_to_length(self.prod_pomdp.observations, max_length=10)
        # self.reward_dict = self.get_reward_dict()
        # self.enumerator = SimpleObservationSequenceEnumerator(self.trans, start_state='0')
        # self.get_reward_tree(max_length=4)

    def get_trans(self):
        trans = {
            '0': {'a': {'1': 0.5, '2': 0.5}, 'b': {'1': 0.5, '2': 0.5}, 'c': {'0': 1}},
            '1': {'a': {'3': 0.5, '4': 0.5}, 'b': {'3': 0.5, '4': 0.5}, 'c': {'1': 1}},
            '2': {'a': {'3': 0.5, '4': 0.5}, 'b': {'3': 0.5, '4': 0.5}, 'c': {'2': 1}},
            '3': {'a': {'5': 1}, 'b': {'1': 0.9, '2': 0.1}, 'c': {'3': 1}},
            '4': {'a': {'5': 1}, 'b': {'1': 0.1, '2': 0.9}, 'c': {'4': 1}},
            '5': {'a': {'3': 0.9, '4': 0.1}, 'b': {'3': 0.1, '4': 0.9}, 'c': {'5': 1}}
        }
        return trans

    def get_observable_operator(self):
        oo_dict = {}
        for obs_t in self.prod_pomdp.observations:
            oo_dict[obs_t] = {}
            for act_t in self.prod_pomdp.actions:
                oo = np.zeros([self.prod_pomdp.state_size, self.prod_pomdp.state_size])
                for i in range(self.prod_pomdp.state_size):
                    for j in range(self.prod_pomdp.state_size):
                        st = self.prod_pomdp.states[i]
                        st_prime = self.prod_pomdp.states[j]
                        # The definition of observable operators
                        # print(st_prime, act_t)
                        if st in self.prod_pomdp.next_supp[st_prime][act_t]:
                            oo[i, j] = self.prod_pomdp.transition[st_prime][act_t][st] * \
                                       self.prod_pomdp.emiss[st_prime][act_t][
                                           obs_t]
                oo_dict[obs_t][act_t] = oo
        return oo_dict

    def p_obs_g_actions(self, y, a_list, observable_operator):
        # Get the real sensing actions
        act_list = [self.prod_pomdp.actions[a] for a in a_list]
        # Give value to the initial state
        mu_0 = self.prod_pomdp.initial_dist
        # Obtain observable operators
        # print(y[-1])
        oo = observable_operator[y[-1]][act_list[-1]]
        # Create a vector with all elements equals to 1
        one_vec = np.ones([1, self.prod_pomdp.state_size])
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

    def p_vtp1_obs_g_actions(self, v_tp1, y, a_list, observable_operator):
        # Get the real sensing actions
        act_list = [self.prod_pomdp.actions[a] for a in a_list]
        # Give value to the initial state
        mu_0 = self.prod_pomdp.initial_dist
        # Obtain observable operators
        oo = observable_operator[y[-1]][act_list[-1]]
        # Create a one-hot vecto which has a 1 element at state index v_{t+1}
        one_hot = np.zeros([1, self.prod_pomdp.state_size])
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

    def p_zT_g_y(self, y, a_list, observable_operator):
        p_obs_g_acts = self.p_obs_g_actions(y, a_list, observable_operator)
        act_T = self.prod_pomdp.actions[a_list[-1]]
        o_T = y[-1]
        p_zT1 = 0
        for st_T in self.prod_pomdp.secret_states:
            v_T = self.prod_pomdp.states.index(st_T)
            emiss_prob = self.prod_pomdp.emiss[st_T][act_T][o_T]
            p_vtp1_obs_g_acts = self.p_vtp1_obs_g_actions(v_T, y[0:-1], a_list[0:-1], observable_operator)
            p_zT1 += emiss_prob * p_vtp1_obs_g_acts / p_obs_g_acts
        p_zT0 = 1 - p_zT1
        p_wT1 = 0
        for st_T in self.prod_pomdp.goal_states:
            v_T = self.prod_pomdp.states.index(st_T)
            emiss_prob = self.prod_pomdp.emiss[st_T][act_T][o_T]
            p_vtp1_obs_g_acts = self.p_vtp1_obs_g_actions(v_T, y[0:-1], a_list[0:-1], observable_operator)
            p_wT1 += emiss_prob * p_vtp1_obs_g_acts / p_obs_g_acts
        return p_zT1, p_zT0, p_wT1

    def entropy(self, y_list, a_list, observable_operator):
        p_zT1, p_zT0, p_wT1 = self.p_zT_g_y(y_list, a_list, observable_operator)
        # print(p_zT1, p_zT0)
        # log2_p_theta_s0_yk = np.log2(p_theta_s0_yk) if p_theta_s0_yk > 0 else 0
        temp_H_1 = p_zT1 * np.log2(p_zT1) if p_zT1 > 0 else 0
        temp_H_0 = p_zT0 * np.log2(p_zT0) if p_zT0 > 0 else 0
        H = - (temp_H_1 + temp_H_0)
        return H, p_wT1

    def reward_function(self, y_list, a_list):
        # a_list = []
        # for i in range(len(y_list)):
        #     a = choice(range(len(self.prod_pomdp.selectable_actions)))  # uniformly random policy
        #     a_list.append(a)
        # For different length of observations, calculate the entropy difference
        if len(y_list) < 2:
            return 0, 0
        elif len(y_list) == 2:
            return self.entropy(y_list, a_list, self.observable_operators)
        else:
            entropy_before, p_wT1_before = self.entropy(y_list[0:-1], a_list[0:-1], self.observable_operators)
            entropy_now, p_wT1_now = self.entropy(y_list, a_list, self.observable_operators)
            return (entropy_before - entropy_now), (p_wT1_before - p_wT1_now)

    def all_sequences_up_to_length(self, elements, max_length):
        """
        Generate all possible sequences from length 1 to max_length.
        Elements can be repeated in sequences.

        Args:
            elements: List of elements to use
            max_length: Maximum length of sequences (T)

        Returns:
            Dictionary with length as key and list of sequences as value
        """
        all_sequences = []

        for length in range(1, max_length + 1):
            sequences = [list(seq) for seq in product(elements, repeat=length)]
            all_sequences += sequences

        return all_sequences


if __name__ == "__main__":
    info_reward = InformationRewards()
