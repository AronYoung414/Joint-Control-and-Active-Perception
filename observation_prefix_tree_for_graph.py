class ObservationTrieNode:
    """Node in the observation sequence prefix tree"""

    def __init__(self):
        self.children = {}
        self.value = None
        self.is_leaf = False


class SimpleObservationSequenceEnumerator:
    """Enumerates all possible observation sequences and assigns values"""

    def __init__(self, transition_dict, start_state='0'):
        self.transitions = transition_dict
        self.start_state = start_state
        self.trie = self._create_prefix_tree()

        # Get all states from transition dictionary
        self.states = set(transition_dict.keys())

        # Define possible observations: all states + null observation
        self.observations = list(self.states) + ['n']

        print(f"Observation enumerator initialized:")
        print(f"  States: {sorted(self.states)}")
        print(f"  Observations: {sorted(self.observations)}")
        print(f"  Starting state: {start_state}")

    def _create_prefix_tree(self):
        """Create the prefix tree structure"""

        class PrefixTree:
            def __init__(self):
                self.root = ObservationTrieNode()

            def insert(self, observation_sequence, value):
                current = self.root
                for obs in observation_sequence:
                    obs_key = str(obs)
                    if obs_key not in current.children:
                        current.children[obs_key] = ObservationTrieNode()
                    current = current.children[obs_key]
                current.is_leaf = True
                current.value = value

            def search(self, observation_sequence):
                current = self.root
                for obs in observation_sequence:
                    obs_key = str(obs)
                    if obs_key not in current.children:
                        return None
                    current = current.children[obs_key]
                return current.value if current.is_leaf else None

            def get_all_sequences(self):
                """Get all sequences stored in the tree"""
                results = []
                self._collect_all_sequences(self.root, [], results)
                return results

            def _collect_all_sequences(self, node, current_sequence, results):
                if node.is_leaf:
                    results.append((current_sequence.copy(), node.value))

                for obs_key, child_node in node.children.items():
                    obs = obs_key  # observations are stored as strings
                    current_sequence.append(obs)
                    self._collect_all_sequences(child_node, current_sequence, results)
                    current_sequence.pop()

            def print_tree_stats(self):
                sequences = self.get_all_sequences()
                print(f"Total sequences stored: {len(sequences)}")
                if sequences:
                    lengths = [len(seq) for seq, _ in sequences]
                    print(f"Sequence length range: {min(lengths)} to {max(lengths)}")

        return PrefixTree()

    def get_possible_transitions(self, state, action):
        """Get all possible next states for a given state-action pair"""
        if state not in self.transitions:
            return []
        if action not in self.transitions[state]:
            return []
        return list(self.transitions[state][action].keys())

    def get_available_actions(self, state):
        """Get all available actions from a given state"""
        if state not in self.transitions:
            return []
        return list(self.transitions[state].keys())

    def enumerate_sequences(self, max_length, value_function=None):
        """
        Enumerate all possible observation sequences up to max_length

        Args:
            max_length: Maximum sequence length
            value_function: Function to assign values to sequences (default: sequence length)
        """
        if value_function is None:
            value_function = lambda seq: len(seq)  # Default: use sequence length as value

        def generate_sequences(current_state, sequence, length):
            # Add current sequence to trie
            current_value = value_function(sequence)
            self.trie.insert(sequence.copy(), current_value)

            # If we've reached max length, stop recursion
            if length >= max_length:
                return

            # Try all possible actions from current state
            available_actions = self.get_available_actions(current_state)

            for action in available_actions:
                # Get all possible next states from this action
                next_states = self.get_possible_transitions(current_state, action)

                for next_state in next_states:
                    # For each possible next state, consider all possible observations
                    for obs in self.observations:
                        # Add observation to sequence
                        sequence.append(obs)

                        # Recurse to next state
                        generate_sequences(next_state, sequence, length + 1)

                        # Backtrack (remove observation)
                        sequence.pop()

        # Start enumeration from starting state
        print(f"\nEnumerating observation sequences starting from state {self.start_state}...")
        print(f"Maximum sequence length: {max_length}")

        # Generate initial observations from starting state
        for obs in self.observations:
            initial_sequence = [obs]
            generate_sequences(self.start_state, initial_sequence, 1)

        print("Enumeration complete!")
        self.trie.print_tree_stats()

    def get_sequence_value(self, sequence):
        """Get the value for a specific sequence"""
        return self.trie.search(sequence)

    def get_all_sequences(self):
        """Get all enumerated sequences with their values"""
        return self.trie.get_all_sequences()

    def print_sample_sequences(self, num_samples=15):
        """Print a sample of sequences for inspection"""
        sequences = self.get_all_sequences()
        print(f"\nSample of {min(num_samples, len(sequences))} observation sequences:")
        print("-" * 60)

        # Sort by value in descending order
        sequences.sort(key=lambda x: x[1], reverse=True)

        for i, (seq, value) in enumerate(sequences[:num_samples]):
            seq_str = " → ".join(seq) if seq else "(empty)"
            print(f"{i + 1:2d}. [{seq_str}] → Value: {value}")

    def analyze_sequences(self):
        """Provide analysis of the enumerated sequences"""
        sequences = self.get_all_sequences()

        if not sequences:
            print("No sequences found!")
            return

        print(f"\n=== SEQUENCE ANALYSIS ===")
        print(f"Total sequences: {len(sequences):,}")

        # Length distribution
        length_counts = {}
        for seq, _ in sequences:
            length = len(seq)
            length_counts[length] = length_counts.get(length, 0) + 1

        print(f"\nSequence length distribution:")
        for length in sorted(length_counts.keys()):
            count = length_counts[length]
            print(f"  Length {length}: {count:,} sequences")

        # Value distribution
        values = [value for _, value in sequences]
        print(f"\nValue statistics:")
        print(f"  Min value: {min(values)}")
        print(f"  Max value: {max(values)}")
        print(f"  Average value: {sum(values) / len(values):.2f}")

        # Observation frequency
        from collections import defaultdict
        obs_counts = defaultdict(int)
        for seq, _ in sequences:
            for obs in seq:
                obs_counts[obs] += 1

        print(f"\nObservation frequency:")
        for obs in sorted(obs_counts.keys()):
            if obs == 'n':
                print(f"  Null observation: {obs_counts[obs]:,}")
            else:
                print(f"  State '{obs}': {obs_counts[obs]:,}")


# Example usage and demonstration
if __name__ == "__main__":
    # Define the transition dictionary
    trans = {
        '0': {'a': {'1': 0.5, '2': 0.5}, 'b': {'1': 0.5, '2': 0.5}, 'c': {'0': 1}},
        '1': {'a': {'3': 0.5, '4': 0.5}, 'b': {'3': 0.5, '4': 0.5}, 'c': {'1': 1}},
        '2': {'a': {'3': 0.5, '4': 0.5}, 'b': {'3': 0.5, '4': 0.5}, 'c': {'2': 1}},
        '3': {'a': {'5': 1}, 'b': {'1': 0.9, '2': 0.1}, 'c': {'3': 1}},
        '4': {'a': {'5': 1}, 'b': {'1': 0.1, '2': 0.9}, 'c': {'4': 1}},
        '5': {'a': {'3': 0.9, '4': 0.1}, 'b': {'3': 0.1, '4': 0.9}, 'c': {'5': 1}}
    }

    # Create enumerator starting from state '0'
    enumerator = SimpleObservationSequenceEnumerator(trans, start_state='0')


    # Define a custom value function
    def custom_value_function(sequence):
        # Base value
        value = 0

        # Add points for each observation
        for obs in sequence:
            if obs == '5':
                value += 10  # High value for observing state 5
            elif obs in ['3', '4']:
                value += 5  # Medium value for observing states 3,4
            elif obs in ['1', '2']:
                value += 2  # Low value for observing states 1,2
            elif obs == '0':
                value += 1  # Minimal value for observing state 0
            # Null observations ('n') add 0 value

        # Bonus for longer sequences
        value += len(sequence) * 0.5

        return value


    # Enumerate sequences up to length 4
    print("=== ENUMERATING OBSERVATION SEQUENCES ===")
    enumerator.enumerate_sequences(max_length=4, value_function=custom_value_function)

    # Print analysis
    enumerator.analyze_sequences()

    # Show sample sequences
    enumerator.print_sample_sequences(20)

    # Test searching for specific sequences
    print(f"\n=== TESTING SEQUENCE LOOKUP ===")
    test_sequences = [
        ['0'],
        ['n'],
        ['5', '5'],
        ['0', '1', '3', '5'],
        ['n', 'n', 'n'],
        ['1', '3', '5'],
    ]

    for seq in test_sequences:
        value = enumerator.get_sequence_value(seq)
        print(f"Sequence {seq}: Value = {value}")

    print(f"\n=== EXAMPLE: DIFFERENT VALUE FUNCTIONS ===")

    # Try with different value functions
    print("1. Length-based values:")
    enumerator2 = SimpleObservationSequenceEnumerator(trans, start_state='0')
    enumerator2.enumerate_sequences(max_length=3, value_function=lambda seq: len(seq))
    print(f"   Sequence ['0', '1'] has value: {enumerator2.get_sequence_value(['0', '1'])}")

    print("\n2. Null-observation penalty:")


    def penalize_nulls(seq):
        return len(seq) - seq.count('n') * 2  # Subtract 2 for each null observation


    enumerator3 = SimpleObservationSequenceEnumerator(trans, start_state='0')
    enumerator3.enumerate_sequences(max_length=3, value_function=penalize_nulls)
    print(f"   Sequence ['0', 'n'] has value: {enumerator3.get_sequence_value(['0', 'n'])}")
    print(f"   Sequence ['0', '1'] has value: {enumerator3.get_sequence_value(['0', '1'])}")