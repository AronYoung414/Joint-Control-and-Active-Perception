from itertools import product
from random import choices

import numpy as np
from pomdp import pomdp
from DFA import DFA

pomdp = pomdp()
dfa = DFA()


class prod_pomdp:

    def __init__(self):
        # Define states
        self.sink_states = ['sink1', 'sink2', 'sink3']
        self.states = [(pomdp_st, dfa_st) for pomdp_st, dfa_st in product(pomdp.states, dfa.states)] + self.sink_states
        self.state_indices = list(range(len(self.states)))
        self.state_size = len(self.states)
        # # Goals
        self.secret_states = ['sink2', 'sink3']
        self.goal_states = ['sink1', 'sink3']
        # self.goals1 = [(pomdp_st, dfa_st) for pomdp_st, dfa_st in product(pomdp.states, dfa.goals1)]
        # self.goals2 = [(pomdp_st, dfa_st) for pomdp_st, dfa_st in product(pomdp.states, dfa.goals2)]
        # Define initial state
        self.initial_states = [(initial_state, dfa.initial_state) for initial_state in pomdp.initial_states]
        self.initial_dist = self.get_initial_distribution()
        self.initial_dist_sampling = [1 / len(self.initial_states) for initial_state in pomdp.initial_states]
        # Define actions
        self.actions = pomdp.actions + ['e']
        self.action_size = len(self.actions)
        self.action_indices = list(range(len(self.actions)))
        # transition probability dictionary
        self.next_supp = self.get_next_supp_with_action()
        self.transition = self.get_transition()
        self.check_the_transition()
        # Define UAV with sensors
        self.obs_noise = pomdp.obs_noise # the noise of sensors
        # Define observations
        self.observations = pomdp.observations
        self.obs_dict = self.get_observation_dictionary()
        self.emiss = self.get_emission_function()
        self.check_emission_function()

    def get_next_supp_with_action(self):
        next_supp = {}
        for st in self.states:
            next_supp[st] = {}
            for act in pomdp.actions:
                if st in self.sink_states:
                    next_supp[st][act] = [st]
                elif st[1] == 0:
                    next_supp[st][act] = ['sink1']
                elif st[1] == 4:
                    next_supp[st][act] = ['sink3']
                else:
                    input_index = dfa.input_symbols.index(pomdp.label_func[st[0]])
                    dfa_st_prime = dfa.transition[st[1]][input_index]
                    next_supp[st][act] = [(pomdp_st_prime, dfa_st_prime) for pomdp_st_prime in pomdp.next_supp[st[0]][act]]
            # discuss the situation of ending action separately
            act = 'e'
            if st[1] == 2:
                next_supp[st][act] = ['sink2']
            else:
                next_supp[st][act] = [st]
        return next_supp

    def get_transition(self):
        trans = {}
        for st in self.states:
            trans[st] = {}
            for act in pomdp.actions:
                trans[st][act] = {}
                for st_prime in self.next_supp[st][act]:
                    if st in self.sink_states:
                        trans[st][act][st_prime] = 1
                    elif st[1] == 0:
                        trans[st][act][st_prime] = 1
                    elif st[1] == 4:
                        trans[st][act][st_prime] = 1
                    else:
                        trans[st][act][st_prime] = pomdp.transition[st[0]][act][st_prime[0]]
            # discuss the situation of ending action separately
            act = 'e'
            trans[st][act] = {}
            for st_prime in self.next_supp[st][act]:
                trans[st][act][st_prime] = 1
        return trans

    def check_the_transition(self):
        for st in self.states:
            for act in self.actions:
                prob = 0
                for st_prime in self.next_supp[st][act]:
                    prob += self.transition[st][act][st_prime]
                if prob != 1:
                    print("The transition is invalid.")
        return 0

    def get_observation_dictionary(self):
        obs_dict = {}
        for st in self.states:
            obs_dict[st] = {}
            for act in pomdp.actions:
                if st in self.sink_states:
                    obs_dict[st][act] = ['n']
                else:
                    obs_dict[st][act] = pomdp.obs_dict[st[0]][act]
            act = 'e'
            obs_dict[st][act] = ['n']
        return obs_dict

    def get_emission_function(self):
        emiss = {}
        for st in self.states:
            emiss[st] = {}
            for act in pomdp.actions:
                emiss[st][act] = {}
                for obs in self.observations:
                    if obs in self.obs_dict[st][act]:
                        if st in self.sink_states:
                            emiss[st][act][obs] = 1
                        else:
                            emiss[st][act][obs] = pomdp.emiss[st[0]][act][obs]
                    else:
                        emiss[st][act][obs] = 0
            act = 'e'
            emiss[st][act] = {}
            for obs in self.observations:
                if obs in self.obs_dict[st][act]:
                    emiss[st][act][obs] = 1
                else:
                    emiss[st][act][obs] = 0
        return emiss

    def check_emission_function(self):
        for st in self.states:
            for act in self.actions:
                prob = 0
                for obs in self.observations:
                    prob += self.emiss[st][act][obs]
                if prob != 1:
                    print("The transition is invalid.")
        return 0

    def get_initial_distribution(self):
        mu_0 = np.zeros([self.state_size, 1])
        for initial_st in self.initial_states:
            s_0 = self.states.index(initial_st)
            mu_0[s_0, 0] = 1 / len(self.initial_states)
        return mu_0

    def next_state_sampler(self, st, act):
        next_supp = self.next_supp[st][act]
        next_prob = [self.transition[st][act][st_prime] for st_prime in next_supp]
        next_state = choices(next_supp, next_prob, k=1)[0]
        return next_state

    def observation_function_sampler(self, st, act):
        observation_set = self.obs_dict[st][act]
        if len(observation_set) == 1:
            return self.observations[-1]
        else:
            return choices(observation_set, [1 - self.obs_noise, self.obs_noise], k=1)[0]