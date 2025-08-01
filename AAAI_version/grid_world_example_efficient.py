from random import choices
import numpy as np


class Environment:

    def __init__(self, width=8, height=8):
        # The width and height of the grid world
        self.width = width
        self.height = height
        # Different parameters for nominal agent and adversary agent (only affects ground robot)
        self.stoPar_n = 0.3  # Lower noise for nominal ground robot (more predictable)
        self.stoPar_a = 0.7  # Higher noise for adversary ground robot (more unpredictable)
        # UAV has its own independent stochasticity parameter
        self.stoPar_uav = 0.4  # Independent noise level for UAV
        # parameter which controls observation noise
        self.obs_noise = 0.5
        # Define obstacles (scale with grid size)
        self.obstacles = [(0, 1), (5, 1), (2, 2), (6, 3), (4, 4), (1, 5), (6, 5), (3, 7)] if width <= 7 else self._generate_obstacles()
        # Define states
        self.whole_states = [(i, j) for i in range(self.width) for j in range(self.height)]
        self.gr_states = list(set(self.whole_states) - set(self.obstacles))
        self.uav_states = self.gr_states
        self.gr_state_indices = list(range(len(self.gr_states)))
        # Define initial state
        self.gr_initial_state = (4, min(7, height - 1))
        self.uav_initial_state = (min(7, width - 1), 4)
        # Define actions
        self.gr_actions = [(1, 0), (-1, 0), (0, 1), (0, -1), (0, 0)]
        self.uav_actions = self.gr_actions
        # Same goal for both agents (they have the same objective)
        self.shared_goal = [(min(4, width - 2), min(0, height - 2))]
        self.goals_n = self.shared_goal  # The goal of nominal agent
        self.goals_a = self.shared_goal  # The goal of adversary agent (same as nominal)
        self.uav_goal = [(0, min(5, height - 2))]
        # reward dictionary
        self.goal_reward = 1
        self.rewards_n = self.get_rewards(self.goals_n)  # The rewards of nominal agent
        self.rewards_a = self.get_rewards(self.goals_a)  # The rewards of adversary (same rewards)
        # discounting factor
        self.gamma = 0.9
        # Compute optimal policies with different transition functions
        self.opt_policy_n = self.value_iteration(self.rewards_n, agent_type='n')[1]  # The optimal policy of nominal agent
        self.opt_policy_a = self.value_iteration(self.rewards_a, agent_type='a')[1]  # The optimal policy of adversary
        # neighbor function (this is O(n²) so acceptable to precompute)
        self.neighbor = self.get_neighbor_function()

    def _generate_obstacles(self):
        """Generate obstacles for larger grids"""
        obstacles = []
        # Add some random obstacles (can be customized)
        for i in range(max(4, self.width // 3)):
            x = np.random.randint(0, self.width)
            y = np.random.randint(0, self.height)
            if (x, y) not in [(0, 0), (self.width - 1, self.height - 1)]:  # Keep start/end clear
                obstacles.append((x, y))
        return obstacles

    def get_neighbor_function(self):
        neigh = {}
        for gr_st in self.gr_states:
            neigh[gr_st] = {}
            for uav_st in self.uav_states:
                dist = np.sqrt((gr_st[0] - uav_st[0]) ** 2 + (gr_st[1] - uav_st[1]) ** 2)
                neigh[gr_st][uav_st] = 2 >= dist >= 0
        return neigh

    def get_rewards(self, goals):
        rewards = {}
        for st in self.gr_states:
            rewards[st] = {}
            for act in self.gr_actions:
                rewards[st][act] = self.goal_reward if st in goals else 0
        return rewards

    def complementary_actions(self, act):
        # Use to find out stochastic transitions, if it stays, no stochasticity, if other actions, return possible stochasticity directions.
        if act == (0, 0):
            return []
        elif act[0] == 0:
            return [(1, 0), (-1, 0)]
        else:
            return [(0, 1), (0, -1)]

    def check_inside(self, st, agent):
        if agent == 'g':
            states = self.gr_states
        elif agent == 'u':
            states = self.uav_states
        else:
            raise ValueError('Invalid agent parameter.')
        return st in states

    def get_transition_prob(self, state, action, next_state, agent='g', agent_type='n'):
        """Compute transition probability on-the-fly with different stoPar for different agent types"""
        if agent == 'g':
            states = self.gr_states
        elif agent == 'u':
            states = self.uav_states
        else:
            raise ValueError('Invalid agent parameter.')

        if state not in states or next_state not in states:
            return 0.0

        if action == (0, 0):  # Stay action
            return 1.0 if state == next_state else 0.0

        # Choose stoPar based on agent and agent_type
        if agent == 'g':  # Ground robot
            if agent_type == 'n':
                stoPar = self.stoPar_n  # Nominal ground robot (less noisy)
            elif agent_type == 'a':
                stoPar = self.stoPar_a  # Adversary ground robot (more noisy)
            else:
                raise ValueError('Invalid agent_type parameter. Use "n" for nominal or "a" for adversary.')
        elif agent == 'u':  # UAV
            stoPar = self.stoPar_uav  # UAV has its own independent stoPar, regardless of agent_type

        prob = 0.0

        # Main intended transition
        tempst = tuple(np.array(state) + np.array(action))
        if self.check_inside(tempst, agent):
            if next_state == tempst:
                prob += 1 - 2 * stoPar
        else:
            if next_state == state:
                prob += 1 - 2 * stoPar

        # Stochastic transitions
        for act_ in self.complementary_actions(action):
            tempst_ = tuple(np.array(state) + np.array(act_))
            if self.check_inside(tempst_, agent):
                if next_state == tempst_:
                    prob += stoPar
            else:
                if next_state == state:
                    prob += stoPar

        return prob

    def get_possible_next_states(self, state, action, agent='g', agent_type='n'):
        """Get all possible next states with non-zero probability
        Note: agent_type only affects ground robot ('g'), UAV ('u') is independent"""
        if agent == 'g':
            states = self.gr_states
        elif agent == 'u':
            states = self.uav_states
        else:
            raise ValueError('Invalid agent parameter.')

        if action == (0, 0):
            return [state]

        possible_states = set()

        # Main transition
        tempst = tuple(np.array(state) + np.array(action))
        if self.check_inside(tempst, agent):
            possible_states.add(tempst)
        else:
            possible_states.add(state)

        # Stochastic transitions
        for act_ in self.complementary_actions(action):
            tempst_ = tuple(np.array(state) + np.array(act_))
            if self.check_inside(tempst_, agent):
                possible_states.add(tempst_)
            else:
                possible_states.add(state)

        return list(possible_states)

    def sample_next_state(self, state, action, agent='g', agent_type='n'):
        """Sample next state without storing full transition matrix
        Note: agent_type only affects ground robot ('g'), UAV ('u') is independent"""
        possible_states = self.get_possible_next_states(state, action, agent, agent_type)
        probabilities = [self.get_transition_prob(state, action, next_st, agent, agent_type)
                         for next_st in possible_states]
        return choices(possible_states, probabilities, k=1)[0]

    def extract_policy(self, rewards, V, agent_type='n'):
        """
        Extracts the optimal policy given the optimal value function.
        """
        policy = {}
        for st in self.gr_states:
            best_action = max(self.gr_actions, key=lambda act:
            rewards[st][act] + self.gamma * sum(
                self.get_transition_prob(st, act, st_prime, 'g', agent_type) * V[st_prime]
                for st_prime in self.get_possible_next_states(st, act, 'g', agent_type)
            ))
            policy[st] = {}
            for act in self.gr_actions:
                policy[st][act] = 1 if act == best_action else 0
        return policy

    def value_iteration(self, rewards, agent_type='n', theta=1e-6):
        """
        Performs Value Iteration to compute the optimal state values with agent-specific transitions.
        """
        V = {st: 0 for st in self.gr_states}  # Initialize value function to zero

        while True:
            delta = 0
            for st in self.gr_states:
                v = V[st]
                V[st] = max(
                    rewards[st][act] + self.gamma * sum(
                        self.get_transition_prob(st, act, st_prime, 'g', agent_type) * V[st_prime]
                        for st_prime in self.get_possible_next_states(st, act, 'g', agent_type))
                    for act in self.gr_actions
                )
                delta = max(delta, abs(v - V[st]))
            if delta < theta:
                break

        return V, self.extract_policy(rewards, V, agent_type)

    def get_policy_transition_prob(self, state, next_state, policy_type='n'):
        """Get transition probability under optimal policy without storing full matrix"""
        if policy_type == 'n':
            policy = self.opt_policy_n
            agent_type = 'n'
        elif policy_type == 'a':
            policy = self.opt_policy_a
            agent_type = 'a'
        else:
            raise ValueError('Invalid policy type')

        total_prob = 0.0
        for action in self.gr_actions:
            action_prob = policy[state][action]
            if action_prob > 0:
                trans_prob = self.get_transition_prob(state, action, next_state, 'g', agent_type)
                total_prob += action_prob * trans_prob

        return total_prob

    def get_policy_possible_next_states(self, state, policy_type='n'):
        """Get all possible next states under policy"""
        possible_states = set()
        if policy_type == 'n':
            policy = self.opt_policy_n
            agent_type = 'n'
        elif policy_type == 'a':
            policy = self.opt_policy_a
            agent_type = 'a'
        else:
            raise ValueError('Invalid policy type')

        for action in self.gr_actions:
            if policy[state][action] > 0:
                possible_states.update(self.get_possible_next_states(state, action, 'g', agent_type))

        return list(possible_states)