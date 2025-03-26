from itertools import product
from graph_example import self

graph = self()


class pomdp:

    def __init__(self):
        # Define states
        # Here tau = 0 (L(s) = !t) means nominal agent and tau = 1 (L(s) = t) means adversary
        self.states = [(gr_st, uav_st, tau) for gr_st, uav_st, tau in
                       product(graph.gr_states, graph.uav_states, [0, 1])]
        # self.state_indices = list(range(len(self.states)))
        # self.state_size = len(self.states)
        # Define initial state
        self.initial_states = [(graph.gr_initial_state, graph.uav_initial_state, 0),
                               (graph.gr_initial_state, graph.uav_initial_state, 1)]
        # self.initial_state_idx = self.states.index(self.initial_state)
        # Define actions
        self.actions = graph.uav_actions
        self.action_size = len(self.actions)
        self.action_indices = list(range(len(self.actions)))
        # transition probability dictionary
        self.next_supp = self.get_next_supp_with_action()
        self.transition = self.get_transition()
        self.check_the_transition()
        # Define UAV with sensors
        self.obs_noise = 0.1  # the noise of sensors
        # Define observations
        self.observations = ['0', '1', '2', '3', '4', '5', 'n']
        self.obs_dict = self.get_observation_dictionary()
        self.emiss = self.get_emission_function()
        self.check_emission_function()
        # Define the atomic propositions
        self.atom_prop = ['t', 'a', 'p']  # a for goal, p for capture
        # Define the labeling function
        self.label_func = self.get_label_function()

    def get_next_supp_with_action(self):
        next_supp = {}
        for st in self.states:
            next_supp[st] = {}
            for act in self.actions:
                if st[2] == 0:
                    gr_next_supp = list(graph.transition_n[st[0]].keys())
                elif st[2] == 1:
                    gr_next_supp = list(graph.transition_a[st[0]].keys())
                else:
                    raise ValueError('Invalid type value.')
                uav_next_supp = list(graph.uav_transition[st[1]][act].keys())
                next_supp[st][act] = [(gr_st, uav_st, st[2]) for gr_st, uav_st in product(gr_next_supp, uav_next_supp)]
        return next_supp

    def get_transition(self):
        trans = {}
        for st in self.states:
            trans[st] = {}
            for act in self.actions:
                trans[st][act] = {}
                for st_prime in self.next_supp[st][act]:
                    if st[2] == 0:
                        trans[st][act][st_prime] = graph.transition_n[st[0]][st_prime[0]] * \
                                                   graph.uav_transition[st[1]][act][st_prime[1]]
                    elif st[2] == 1:
                        trans[st][act][st_prime] = graph.transition_a[st[0]][st_prime[0]] * \
                                                   graph.uav_transition[st[1]][act][st_prime[1]]
                    else:
                        raise ValueError('Invalid type value.')
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
            for act in self.actions:
                if graph.neighbor[st[0]][st[1]]:
                    obs_dict[st][act] = [st[0], 'n']
                else:
                    obs_dict[st][act] = ['n']
        return obs_dict

    def get_emission_function(self):
        emiss = {}
        for st in self.states:
            emiss[st] = {}
            for act in self.actions:
                emiss[st][act] = {}
                for obs in self.observations:
                    if obs in self.obs_dict[st][act]:
                        if graph.neighbor[st[0]][st[1]]:
                            if st[0] == obs:
                                emiss[st][act][obs] = 1 - self.obs_noise
                            else:
                                emiss[st][act][obs] = self.obs_noise
                        else:
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

    def get_label_function(self):
        lab = {}
        for st in self.states:
            lab[st] = []
            if st[2] == 1:
                lab[st] += self.atom_prop[0]
            if st[1] in graph.uav_goal:
                lab[st] += self.atom_prop[1]
            if st[0] == st[1]:
                lab[st] += self.atom_prop[2]
        return lab
