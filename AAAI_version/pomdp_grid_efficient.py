from itertools import product
from random import choices


# Assuming the revised Environment class is imported
# from grid_world_example import Environment

class POMDP:

    def __init__(self, grid_world):
        self.grid_world = grid_world
        # The width and height of the grid world
        self.width = grid_world.width
        self.height = grid_world.height
        # Define states
        # Here tau = 0 (L(s) = !t) means nominal agent and tau = 1 (L(s) = t) means adversary
        self.states = [(gr_st, uav_st, tau) for gr_st, uav_st, tau in
                       product(grid_world.gr_states, grid_world.uav_states, [0, 1])]
        # Define initial state
        self.initial_states = [(grid_world.gr_initial_state, grid_world.uav_initial_state, 0),
                               (grid_world.gr_initial_state, grid_world.uav_initial_state, 1)]
        # Define actions
        self.actions = grid_world.uav_actions
        self.action_size = len(self.actions)
        self.action_indices = list(range(len(self.actions)))
        # Define UAV with sensors
        self.obs_noise = grid_world.obs_noise  # the noise of sensors
        # Define observations
        self.observations = [(gr_st, uav_st) for gr_st, uav_st in
                             product((grid_world.gr_states + ['n']), grid_world.uav_states)]
        # Define the atomic propositions
        self.atom_prop = ['t', 'a', 'p']  # a for goal, p for capture, t indicates adversary
        # Define the labeling function
        self.label_func = self.get_label_function()

    def get_possible_next_states(self, state, action):
        """Get all possible next states with non-zero probability"""
        gr_st, uav_st, tau = state

        # Ground robot transitions depend on agent type (tau)
        if tau == 0:
            gr_next_states = self.grid_world.get_policy_possible_next_states(gr_st, 'n')
        elif tau == 1:
            gr_next_states = self.grid_world.get_policy_possible_next_states(gr_st, 'a')
        else:
            raise ValueError('Invalid tau value.')

        # UAV transitions are independent of tau (agent_type doesn't matter for UAV)
        uav_next_states = self.grid_world.get_possible_next_states(uav_st, action, 'u', 'n')

        return [(gr_st_next, uav_st_next, tau)
                for gr_st_next, uav_st_next in product(gr_next_states, uav_next_states)]

    def get_transition_prob(self, state, action, next_state):
        """Compute transition probability on-the-fly"""
        gr_st, uav_st, tau = state
        gr_st_next, uav_st_next, tau_next = next_state

        # Tau doesn't change in transitions
        if tau != tau_next:
            return 0.0

        # Ground robot transition probability depends on agent type (tau)
        if tau == 0:
            gr_prob = self.grid_world.get_policy_transition_prob(gr_st, gr_st_next, 'n')
        elif tau == 1:
            gr_prob = self.grid_world.get_policy_transition_prob(gr_st, gr_st_next, 'a')
        else:
            raise ValueError('Invalid tau value.')

        # UAV transition probability is independent of tau (uses its own stoPar_uav)
        uav_prob = self.grid_world.get_transition_prob(uav_st, action, uav_st_next, 'u', 'n')

        return gr_prob * uav_prob

    def sample_next_state(self, state, action):
        """Sample next state without storing full transition matrix"""
        possible_states = self.get_possible_next_states(state, action)
        probabilities = [self.get_transition_prob(state, action, next_st)
                         for next_st in possible_states]
        return choices(possible_states, probabilities, k=1)[0]

    def get_possible_observations(self, state, action):
        """Get all possible observations for a state-action pair"""
        gr_st, uav_st, tau = state

        if self.grid_world.neighbor[gr_st][uav_st]:
            return [(gr_st, uav_st), ('n', uav_st)]
        else:
            return [('n', uav_st)]

    def get_emission_prob(self, state, action, observation):
        """Compute emission probability on-the-fly"""
        gr_st, uav_st, tau = state

        possible_obs = self.get_possible_observations(state, action)

        if observation not in possible_obs:
            return 0.0

        if self.grid_world.neighbor[gr_st][uav_st]:
            if observation == (gr_st, uav_st):
                return 1 - self.obs_noise
            elif observation == ('n', uav_st):
                return self.obs_noise
            else:
                return 0.0
        else:
            if observation == ('n', uav_st):
                return 1.0
            else:
                return 0.0

    def sample_observation(self, state, action):
        """Sample observation without storing full emission matrix"""
        possible_obs = self.get_possible_observations(state, action)
        if len(possible_obs) == 1:
            return possible_obs[0]
        else:
            probabilities = [self.get_emission_prob(state, action, obs)
                             for obs in possible_obs]
            return choices(possible_obs, probabilities, k=1)[0]

    def get_label_function(self):
        lab = {}
        for st in self.states:
            gr_st, uav_st, tau = st
            lab[st] = []
            if tau == 1:
                lab[st].append(self.atom_prop[0])  # 't'
            if uav_st in self.grid_world.uav_goal:
                lab[st].append(self.atom_prop[1])  # 'a'
            if gr_st == uav_st:
                lab[st].append(self.atom_prop[2])  # 'p'
        return lab

    def check_transition_validity(self, num_samples=1000):
        """Verify that transition probabilities sum to 1 for random samples"""
        import random

        for _ in range(num_samples):
            state = random.choice(self.states)
            action = random.choice(self.actions)

            possible_next = self.get_possible_next_states(state, action)
            total_prob = sum(self.get_transition_prob(state, action, next_st)
                             for next_st in possible_next)

            if abs(total_prob - 1.0) > 0.01:
                print(f"Invalid transition probability for {state}, {action}: {total_prob}")
                return False

        print("Transition probabilities are valid")
        return True

    def check_emission_validity(self, num_samples=1000):
        """Verify that emission probabilities sum to 1 for random samples"""
        import random

        for _ in range(num_samples):
            state = random.choice(self.states)
            action = random.choice(self.actions)

            total_prob = sum(self.get_emission_prob(state, action, obs)
                             for obs in self.observations)

            if abs(total_prob - 1.0) > 0.01:
                print(f"Invalid emission probability for {state}, {action}: {total_prob}")
                return False

        print("Emission probabilities are valid")
        return True