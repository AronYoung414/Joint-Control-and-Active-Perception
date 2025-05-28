class ObservationTrieNode:
    """Node in the observation sequence prefix tree"""

    def __init__(self):
        self.children = {}  # Dictionary to store child nodes
        self.value = None  # Value stored at this node (for leaf nodes)
        self.is_leaf = False  # Flag to indicate if this is a leaf node

    def __repr__(self):
        return f"TrieNode(children={list(self.children.keys())}, value={self.value}, is_leaf={self.is_leaf})"


class MovementSequenceEnumerator:
    """Enumerates all possible movement sequences in a 6x6 grid"""

    def __init__(self, start_pos=(0, 3), grid_size=6):
        self.start_pos = start_pos
        self.grid_size = grid_size
        self.trie = self._create_prefix_tree()

        # Movement directions: up, down, left, right, stay
        self.moves = {
            'up': (0, -1),
            'down': (0, 1),
            'left': (-1, 0),
            'right': (1, 0),
            'stay': (0, 0)
        }

    def _create_prefix_tree(self):
        """Create the prefix tree structure"""

        class PrefixTree:
            def __init__(self):
                self.root = ObservationTrieNode()

            def _encode_observation(self, obs):
                if obs == 'n' or obs is None:
                    return 'n'
                elif isinstance(obs, tuple) and len(obs) == 2:
                    x, y = obs
                    return f"{x},{y}"
                else:
                    raise ValueError(f"Invalid observation format: {obs}")

            def insert(self, observation_sequence, value):
                current = self.root
                for obs in observation_sequence:
                    obs_key = self._encode_observation(obs)
                    if obs_key not in current.children:
                        current.children[obs_key] = ObservationTrieNode()
                    current = current.children[obs_key]
                current.is_leaf = True
                current.value = value

            def search(self, observation_sequence):
                current = self.root
                for obs in observation_sequence:
                    obs_key = self._encode_observation(obs)
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
                    if obs_key == 'n':
                        obs = 'n'
                    else:
                        x, y = map(int, obs_key.split(','))
                        obs = (x, y)

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

    def is_valid_position(self, pos):
        """Check if position is within grid bounds"""
        x, y = pos
        return 0 <= x < self.grid_size and 0 <= y < self.grid_size

    def get_next_positions(self, current_pos):
        """Get all valid next positions from current position"""
        x, y = current_pos
        next_positions = []

        for move_name, (dx, dy) in self.moves.items():
            new_pos = (x + dx, y + dy)
            if self.is_valid_position(new_pos):
                next_positions.append((new_pos, move_name))

        return next_positions

    def enumerate_sequences(self, max_length, value_function=None):
        """
        Enumerate all possible movement sequences up to max_length

        Args:
            max_length: Maximum sequence length
            value_function: Function to assign values to sequences (default: sequence length)
        """
        if value_function is None:
            value_function = lambda seq: len(seq)  # Default: use sequence length as value

        def generate_sequences(current_pos, sequence, length):
            # Add current sequence to trie
            current_value = value_function(sequence)
            self.trie.insert(sequence.copy(), current_value)

            # If we've reached max length, stop recursion
            if length >= max_length:
                return

            # Generate all possible next moves
            next_positions = self.get_next_positions(current_pos)

            for next_pos, move_name in next_positions:
                sequence.append(next_pos)
                generate_sequences(next_pos, sequence, length + 1)
                sequence.pop()  # Backtrack

        # Start enumeration from starting position
        print(f"Enumerating sequences starting from {self.start_pos}...")
        print(f"Maximum sequence length: {max_length}")
        print(f"Grid size: {self.grid_size}x{self.grid_size}")

        initial_sequence = [self.start_pos]
        generate_sequences(self.start_pos, initial_sequence, 1)

        print("Enumeration complete!")
        self.trie.print_tree_stats()

    def get_sequence_value(self, sequence):
        """Get the value for a specific sequence"""
        return self.trie.search(sequence)

    def get_all_sequences(self):
        """Get all enumerated sequences with their values"""
        return self.trie.get_all_sequences()

    def print_sample_sequences(self, num_samples=10):
        """Print a sample of sequences for inspection"""
        sequences = self.get_all_sequences()
        print(f"\nSample of {min(num_samples, len(sequences))} sequences:")
        print("-" * 60)

        for i, (seq, value) in enumerate(sequences[:num_samples]):
            moves = []
            for j in range(1, len(seq)):
                prev_pos = seq[j - 1]
                curr_pos = seq[j]

                if curr_pos == prev_pos:
                    moves.append("stay")
                else:
                    dx = curr_pos[0] - prev_pos[0]
                    dy = curr_pos[1] - prev_pos[1]

                    if dx == 1:
                        moves.append("right")
                    elif dx == -1:
                        moves.append("left")
                    elif dy == 1:
                        moves.append("down")
                    elif dy == -1:
                        moves.append("up")

            move_str = " → ".join(moves) if moves else "start"
            print(f"{i + 1:2d}. {seq} | Moves: {move_str} | Value: {value}")

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


# Example usage and demonstration
if __name__ == "__main__":
    # Create enumerator starting from (0, 3)
    enumerator = MovementSequenceEnumerator(start_pos=(0, 3))


    # Define a custom value function (example: distance from origin + sequence length)
    def custom_value_function(sequence):
        if len(sequence) == 0:
            return 0

        # Get final position
        final_pos = sequence[-1]

        # Calculate Manhattan distance from origin (0,0)
        distance = abs(final_pos[0]) + abs(final_pos[1])

        # Combine distance and sequence length
        return distance * 2 + len(sequence)


    # Enumerate sequences up to length 4 (to keep it manageable for demonstration)
    print("=== ENUMERATING MOVEMENT SEQUENCES ===")
    enumerator.enumerate_sequences(max_length=4, value_function=custom_value_function)

    # Print analysis
    enumerator.analyze_sequences()

    # Show sample sequences
    enumerator.print_sample_sequences(15)

    # Test searching for specific sequences
    print(f"\n=== TESTING SEQUENCE LOOKUP ===")
    test_sequences = [
        [(0, 3)],  # Starting position
        [(0, 3), (0, 2)],  # Move up
        [(0, 3), (1, 3)],  # Move right
        [(0, 3), (0, 3)],  # Stay in place
        [(0, 3), (0, 2), (0, 1), (0, 0)],  # Move up 3 times
    ]

    for seq in test_sequences:
        value = enumerator.get_sequence_value(seq)
        print(f"Sequence {seq}: Value = {value}")

    print(f"\n=== MOVEMENT POSSIBILITIES FROM (0,3) ===")
    next_moves = enumerator.get_next_positions((0, 3))
    print(f"From (0,3), can move to:")
    for pos, move in next_moves:
        print(f"  {pos} ({move})")