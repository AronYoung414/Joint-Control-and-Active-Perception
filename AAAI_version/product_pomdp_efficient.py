from itertools import product
from random import choices
import numpy as np


# Assuming DFA class exists and is imported
# from DFA import DFA

class ProductPOMDP:

    def __init__(self, pomdp, dfa):
        self.pomdp = pomdp
        self.dfa = dfa

        # The width and height of the grid world
        self.width = pomdp.width
        self.height = pomdp.height

        # Define states
        self.sink_states = ['sink1', 'sink2', 'sink3']
        self.regular_states = [(pomdp_st, dfa_st) for pomdp_st, dfa_st in
                               product(self.pomdp.states, self.dfa.states)]
        self.states = self.regular_states + self.sink_states
        self.state_indices = list(range(len(self.states)))
        self.state_size = len(self.states)

        # Goals
        self.secret_states = ['sink2', 'sink3']
        self.goal_states = ['sink1', 'sink3']

        # Define initial state
        self.initial_states = [(initial_state, self.dfa.initial_state)
                               for initial_state in self.pomdp.initial_states]
        self.initial_dist = self.get_initial_distribution()
        self.initial_dist_sampling = [1 / len(self.initial_states)
                                      for _ in self.pomdp.initial_states]

        # Define actions
        self.actions = self.pomdp.actions + ['e']
        self.selectable_actions = self.pomdp.actions
        self.action_size = len(self.actions)
        self.action_indices = list(range(len(self.actions)))

        # Define UAV with sensors
        self.obs_noise = self.pomdp.obs_noise

        # Define observations
        self.observations = self.pomdp.observations + [('n', 'n')]

    def get_possible_next_states(self, state, action):
        """Get all possible next states with non-zero probability"""
        if state in self.sink_states:
            return [state]

        pomdp_st, dfa_st = state

        if action == 'e':
            if dfa_st == 2:  # Specific DFA state for ending
                return ['sink2']  # adversary is not captured
            else:
                return [state]

        # For regular actions
        if dfa_st == 0:
            return ['sink1']  # UAV reaches the goal (nominal agent)
        elif dfa_st == 4:
            return ['sink3']  # adversary is captured
        else:
            # Get label for current POMDP state
            current_label = self.pomdp.label_func[pomdp_st]
            input_index = self.dfa.input_symbols.index(current_label)
            dfa_st_prime = self.dfa.transition[dfa_st][input_index]

            pomdp_next_states = self.pomdp.get_possible_next_states(pomdp_st, action)

            return [(pomdp_st_prime, dfa_st_prime) for pomdp_st_prime in pomdp_next_states]

    def get_transition_prob(self, state, action, next_state):
        """Compute transition probability on-the-fly"""
        if state in self.sink_states:
            return 1.0 if next_state == state else 0.0

        pomdp_st, dfa_st = state

        if action == 'e':
            if dfa_st == 2 and next_state == 'sink2':
                return 1.0
            elif next_state == state:
                return 1.0
            else:
                return 0.0

        # For regular actions
        if dfa_st == 0:
            return 1.0 if next_state == 'sink1' else 0.0
        elif dfa_st == 4:
            return 1.0 if next_state == 'sink3' else 0.0
        else:
            if next_state in self.sink_states:
                return 0.0

            pomdp_st_next, dfa_st_next = next_state

            # Check if DFA transition is valid
            current_label = self.pomdp.label_func[pomdp_st]
            input_index = self.dfa.input_symbols.index(current_label)
            expected_dfa_next = self.dfa.transition[dfa_st][input_index]

            if dfa_st_next != expected_dfa_next:
                return 0.0

            return self.pomdp.get_transition_prob(pomdp_st, action, pomdp_st_next)

    def sample_next_state(self, state, action):
        """Sample next state without storing full transition matrix"""
        possible_states = self.get_possible_next_states(state, action)
        probabilities = [self.get_transition_prob(state, action, next_st)
                         for next_st in possible_states]
        return choices(possible_states, probabilities, k=1)[0]

    def get_possible_observations(self, state, action):
        """Get all possible observations for a state-action pair"""
        if state in self.sink_states or action == 'e':
            return [('n', 'n')]

        pomdp_st, dfa_st = state
        return self.pomdp.get_possible_observations(pomdp_st, action)

    def get_emission_prob(self, state, action, observation):
        """Compute emission probability on-the-fly"""
        possible_obs = self.get_possible_observations(state, action)

        if observation not in possible_obs:
            return 0.0

        if state in self.sink_states or action == 'e':
            return 1.0 if observation == ('n', 'n') else 0.0

        pomdp_st, dfa_st = state
        return self.pomdp.get_emission_prob(pomdp_st, action, observation)

    def sample_observation(self, state, action):
        """Sample observation without storing full emission matrix"""
        possible_obs = self.get_possible_observations(state, action)
        if len(possible_obs) == 1:
            return possible_obs[0]
        else:
            probabilities = [self.get_emission_prob(state, action, obs)
                             for obs in possible_obs]
            return choices(possible_obs, probabilities, k=1)[0]

    def get_initial_distribution(self):
        mu_0 = np.zeros([self.state_size, 1])
        for initial_st in self.initial_states:
            s_0 = self.states.index(initial_st)
            mu_0[s_0, 0] = 1 / len(self.initial_states)
        return mu_0

    def step(self, state, action):
        """Execute one step in the environment"""
        next_state = self.sample_next_state(state, action)
        observation = self.sample_observation(next_state, action)

        # Determine reward (customize based on your reward structure)
        reward = self.get_reward(state, action, next_state)

        return next_state, observation, reward

    def get_reward(self, state, action, next_state):
        """Define reward function - customize as needed"""
        if next_state == 'sink1':  # Goal reached
            return 10.0
        elif next_state == 'sink2':  # Adversary not captured
            return -10.0
        elif next_state == 'sink3':  # Adversary captured
            return 5.0
        else:
            return -0.1  # Small negative reward for each step

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

        print("Product POMDP transition probabilities are valid")
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

        print("Product POMDP emission probabilities are valid")
        return True

    def simulate_episode(self, max_steps=100):
        """Simulate one episode for testing"""
        # Start from random initial state
        current_state = choices(self.initial_states, k=1)[0]
        trajectory = []

        for step in range(max_steps):
            # Choose random action (replace with your policy)
            action = choices(self.selectable_actions, k=1)[0]

            # Execute step
            next_state, observation, reward = self.step(current_state, action)

            trajectory.append({
                'step': step,
                'state': current_state,
                'action': action,
                'next_state': next_state,
                'observation': observation,
                'reward': reward
            })

            current_state = next_state

            # Check if we've reached a terminal state
            if current_state in self.sink_states:
                break

        return trajectory