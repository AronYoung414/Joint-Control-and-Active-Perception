import itertools
import numpy as np

from product_pomdp import prod_pomdp

prod_pomdp = prod_pomdp()


def permutations_with_repetition(elements, length):
    # Generate permutations with repetition
    results = itertools.product(elements, repeat=length)
    # Convert each tuple to a string and collect into a list
    string_results = [''.join(result) for result in results]
    return string_results


class FSC:

    def __init__(self):
        # The length of memory
        self.K = 2
        # The observations (the input of the finite state controller)
        self.observations = prod_pomdp.observations
        self.observations_size = len(self.observations)
        # the memory space
        self.memory_space, self.memory_size = self.get_memory_space()
        # The transition
        self.transition = self.get_transition()

    def get_memory_space(self):
        K = self.K
        memory_space = ['l']
        memory_size = 1  # Add the initial observation
        for k in range(1, K + 1):
            memory_space += permutations_with_repetition(self.observations, k)
            memory_size += self.observations_size ** k
        return memory_space, memory_size

    def get_transition(self):
        trans = {}
        for m in range(self.memory_size):
            trans[m] = {}
            memory = self.memory_space[m]
            for o in range(self.observations_size):
                obs = self.observations[o]
                if memory == 'l':
                    next_memory = obs
                else:
                    if len(memory) < self.K:
                        next_memory = memory + obs
                    else:
                        temp_memory = memory + obs
                        next_memory = temp_memory[1:]
                nm = self.memory_space.index(next_memory)
                trans[m][o] = nm
        return trans
