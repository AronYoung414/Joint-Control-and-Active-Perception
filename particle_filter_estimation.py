import numpy as np
# from product_pomdp import prod_pomdp
# from observation_prefix_tree_for_graph import SimpleObservationSequenceEnumerator
# from random import choices
import random
from matplotlib import pyplot as plt

from information_rewards import InformationRewards


# from itertools import permutations, product
# import pickle


class particle_filter:

    def __init__(self, env, initial_state, num_particles=200, sensor_accuracy=0.9):
        self.num_particles = num_particles
        self.env = env
        # get the initial state
        self.true_state = initial_state
        self.particles = []
        self.weights = []
        self.sensor_accuracy = sensor_accuracy
        self.initialize_particles()

    def initialize_particles(self):
        """Initialize particles uniformly over valid positions"""
        self.particles = []
        self.weights = []

        # Sample particles uniformly
        for _ in range(self.num_particles):
            particle_pos = random.choice(self.env.states)
            self.particles.append(particle_pos)
            self.weights.append(1.0 / self.num_particles)

    def predict(self, act):
        """Prediction step: move all particles according to action"""
        new_particles = []
        for particle in self.particles:
            new_particle = self.env.next_state_sampler(particle, act)
            new_particles.append(new_particle)
        self.particles = new_particles

    def update(self, observation, act):
        """Update step: weight particles based on observation"""
        total_weight = 0

        for i, particle in enumerate(self.particles):
            # Get expected observation for this particle
            expected_obs = self.env.observation_function_sampler(particle, act)

            # Calculate likelihood
            if expected_obs == observation:
                # Correct observation
                self.weights[i] = self.sensor_accuracy
            else:
                # Incorrect observation
                self.weights[i] = (1 - self.sensor_accuracy) / 10  # Small probability

            total_weight += self.weights[i]

        # Normalize weights
        if total_weight > 0:
            self.weights = [w / total_weight for w in self.weights]
        else:
            # If all weights are zero, reset to uniform
            self.weights = [1.0 / self.num_particles] * self.num_particles

    def get_effective_particles(self):
        """Calculate effective sample size"""
        sum_squares = sum(w ** 2 for w in self.weights)
        return 1.0 / sum_squares if sum_squares > 0 else 0

    def resample(self):
        """Systematic resampling"""
        effective_particles = self.get_effective_particles()

        if effective_particles < self.num_particles / 3:
            new_particles = []

            # Systematic resampling
            step = 1.0 / self.num_particles
            start = random.uniform(0, step)

            cumulative_weight = 0
            weight_index = 0

            for i in range(self.num_particles):
                target = start + i * step

                while cumulative_weight < target and weight_index < len(self.weights):
                    cumulative_weight += self.weights[weight_index]
                    weight_index += 1

                if weight_index > 0:
                    new_particles.append(self.particles[weight_index - 1])

            self.particles = new_particles
            self.weights = [1.0 / self.num_particles] * self.num_particles
            return True

        return False

    def get_position_probabilities(self):
        """Get probability distribution over all positions"""
        probs = {}

        # Initialize with zeros for all valid positions
        for st in self.env.states:
            probs[st] = 0.0

        # Add particle weights
        for i, particle in enumerate(self.particles):
            if particle in probs:
                probs[particle] += self.weights[i]

        return probs

    def calculate_entropy(self, obs, act):
        """Calculate entropy of belief distribution"""
        self.predict(act)
        self.update(obs, act)
        self.resample()
        probs = self.get_position_probabilities()
        p_zT1 = 0
        for st_T in self.env.secret_states:
            p_zT1 += probs[st_T]
        p_zT0 = 1 - p_zT1
        temp_H_1 = p_zT1 * np.log2(p_zT1) if p_zT1 > 0 else 0
        temp_H_0 = p_zT0 * np.log2(p_zT0) if p_zT0 > 0 else 0
        H = - (temp_H_1 + temp_H_0)
        p_wT1 = 0
        for st_T in self.env.goal_states:
            p_wT1 += probs[st_T]
        return H, p_wT1

    # def reward_function(self):
    #     # For different length of observations, calculate the entropy difference
    #     if len(self.obs_list) < 1:
    #         return 0, 0
    #     elif len(self.obs_list) == 1:
    #         return self.calculate_entropy()
    #     else:
    #         entropy_before, p_wT1_before = self.calculate_entropy()
    #         entropy_now, p_wT1_now = self.calculate_entropy()
    #         return (entropy_before - entropy_now), (p_wT1_before - p_wT1_now)


def main():
    # initialize
    info_rewards = InformationRewards()
    pp = info_rewards.prod_pomdp
    state = pp.initial_states[0]
    pf = particle_filter(pp, state, num_particles=10000)
    max_steps = 20
    # entropy_before = 0
    entropy_now = 0
    # p_wT1_before = 0
    p_wT1_now = 0
    total_reward_particle = 0
    total_entropy_particle = 0
    total_probs_particle = 0
    total_reward_oo = 0
    total_entropy_oo = 0
    total_probs_oo = 0
    as_list = []
    obs_list = []
    entropy_diff_list_particle = []
    prob_diff_list_particle = []
    reward_list_particle = []
    entropy_diff_list_oo = []
    prob_diff_list_oo = []
    reward_list_oo = []
    for step in range(max_steps):
        # sample random action
        act = random.choice(pp.actions)
        a = pp.actions.index(act)
        as_list.append(a)
        # Obtain observations
        obs = pp.observation_function_sampler(state, act)
        obs_list.append(obs)
        # Obtain entropies
        # particle filter method
        entropy_before = entropy_now
        p_wT1_before = p_wT1_now
        entropy_now, p_wT1_now = pf.calculate_entropy(obs, act)
        # entropy difference
        entropy_diff = entropy_before - entropy_now
        entropy_diff_list_particle.append(entropy_diff)
        # probability difference
        prob_diff = p_wT1_before - p_wT1_now
        prob_diff_list_particle.append(prob_diff)
        # obtain rewards
        reward = entropy_diff - prob_diff
        reward_list_particle.append(reward)
        # Calculate total values
        total_reward_particle += reward
        total_entropy_particle += entropy_diff
        total_probs_particle += prob_diff
        # observable operator method
        # Obtain entropy, probability, and rewards
        entropy_diff, prob_diff = info_rewards.reward_function(obs_list, as_list)
        reward = entropy_diff - prob_diff
        entropy_diff_list_oo.append(entropy_diff)
        prob_diff_list_oo.append(prob_diff)
        reward_list_oo.append(reward)
        # Calculate total values
        total_reward_oo += reward
        total_entropy_oo += entropy_diff
        total_probs_oo += prob_diff
        # next state
        state = pp.next_state_sampler(state, act)
    # ending action
    final_act = 'e'
    final_a = pp.actions.index(final_act)
    as_list.append(final_a)
    final_state = pp.next_state_sampler(state, final_act)
    final_obs = pp.observation_function_sampler(final_state, final_act)
    obs_list.append(final_obs)
    entropy_before = entropy_now
    p_wT1_before = p_wT1_now
    entropy_now, p_wT1_now = pf.calculate_entropy(final_obs, final_act)
    # entropy difference
    entropy_diff = entropy_before - entropy_now
    entropy_diff_list_particle.append(entropy_diff)
    # probability difference
    prob_diff = p_wT1_before - p_wT1_now
    prob_diff_list_particle.append(prob_diff)
    # obtain rewards
    reward = entropy_diff - prob_diff
    reward_list_particle.append(reward)
    # Calculate total values
    total_reward_particle += reward
    total_entropy_particle += entropy_diff
    total_probs_particle += prob_diff
    # Obtain entropy, probability, and rewards
    entropy_diff, prob_diff = info_rewards.reward_function(obs_list, as_list)
    reward = entropy_diff - prob_diff
    entropy_diff_list_oo.append(entropy_diff)
    prob_diff_list_oo.append(prob_diff)
    reward_list_oo.append(reward)
    # Calculate total values
    total_reward_oo += reward
    total_entropy_oo += entropy_diff
    total_probs_oo += prob_diff

    print("The list of entropies difference (pf)", entropy_diff_list_particle)
    print("The list of probability difference (pf)", prob_diff_list_particle)
    print("The list of reward (pf)", reward_list_particle)
    print("Total entropy is (pf)", -total_entropy_particle)
    print("Total probability is (pf)", -total_probs_particle)
    print("Total reward is (pf)", total_reward_particle)

    print("The list of entropies difference (oo)", entropy_diff_list_oo)
    print("The list of probability difference (oo)", prob_diff_list_oo)
    print("The list of reward (oo)", reward_list_oo)
    print("Total entropy is (oo)", -total_entropy_oo)
    print("Total probability is (oo)", -total_probs_oo)
    print("Total reward is (oo)", total_reward_oo)

    fig, axes = plt.subplots(1, 3, figsize=(15, 10))
    fig.suptitle('Particle Filter vs Observable Operator Method Comparison', fontsize=16)

    steps = range(len(entropy_diff_list_oo))

    # Use 1D indexing for the axes
    axes[0].plot(steps, entropy_diff_list_particle, 'b-', label='Particle Filter', alpha=0.7)
    axes[0].plot(steps, entropy_diff_list_oo, 'r--', label='Observable Operator', alpha=0.7)
    axes[0].set_title('Entropy Differences')
    axes[0].set_xlabel('Step')
    axes[0].set_ylabel('Entropy Diff')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(steps, prob_diff_list_particle, 'b-', label='Particle Filter', alpha=0.7)
    axes[1].plot(steps, prob_diff_list_oo, 'r--', label='Observable Operator', alpha=0.7)
    axes[1].set_title('Probability Differences')
    axes[1].set_xlabel('Step')
    axes[1].set_ylabel('Prob Diff')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(steps, reward_list_particle, 'b-', label='Particle Filter', alpha=0.7)
    axes[2].plot(steps, reward_list_oo, 'r--', label='Observable Operator', alpha=0.7)
    axes[2].set_title('Rewards')
    axes[2].set_xlabel('Step')
    axes[2].set_ylabel('Reward')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
