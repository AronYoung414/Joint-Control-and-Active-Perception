from pomdp import pomdp

pomdp = pomdp()


class DFA:

    def __init__(self):
        # Define states
        self.states = [0, 1, 2, 3]
        # Goals
        self.goals = [0]
        # Define initial state
        self.initial_state = 3
        # Define actions
        self.input_symbols = self.get_input_symbols()
        self.input_size = len(self.input_symbols)
        self.input_indices = list(range(len(self.input_symbols)))
        # Define transition
        self.transition = self.get_transition()

    @staticmethod
    def get_input_symbols():
        inputs = set()
        for st in pomdp.states:
            inputs.add(tuple(pomdp.label_func[st]))
        inputs_list = []
        for input in inputs:
            inputs_list.append(list(input))
        return inputs_list

    def get_transition(self):
        trans = {}
        for st in self.states:
            trans[st] = {}
            for i in self.input_indices:
                input = self.input_symbols[i]
                if st == 3:
                    # if len(input) == 0:
                    #     trans[st][i] = st
                    if 'a' not in input and 't' not in input:
                        trans[st][i] = 1
                    elif 'p' not in input and 't' in input:
                        trans[st][i] = 2
                    elif ('a' in input and 't' not in input) or ('p' in input and 't' in input):
                        trans[st][i] = 0
                    else:
                        trans[st][i] = st
                elif st == 1:
                    if 'a' not in input:
                        trans[st][i] = st
                    else:
                        trans[st][i] = 0
                elif st == 2:
                    if 'p' not in input:
                        trans[st][i] = st
                    else:
                        trans[st][i] = 0
                elif st == 0:
                    trans[st][i] = 0
                else:
                    raise ValueError('Invalid automata state.')
        return trans