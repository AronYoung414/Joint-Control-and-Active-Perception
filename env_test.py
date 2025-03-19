from pomdp import pomdp
from DFA import DFA
from product_pomdp import prod_pomdp

pomdp = pomdp()
dfa = DFA()
prod_pomdp = prod_pomdp()

# prod_pomdp.check_the_transition()

print(prod_pomdp.transition[(('0', '0', 0), 0)]['a'][(('1', '1', 0), 0)])

# print(pomdp.transition)
# print(pomdp.emiss)
# print(pomdp.label_func)

# print(dfa.input_symbols)

# for st in dfa.states:
#     for i in dfa.input_indices:
#         input = dfa.input_symbols[i]
#         print('State:', st, 'Input', input, 'Next State', dfa.transition[st][i])

