"""Test for topological sort"""

from pynever import nodes
from pynever.networks import SequentialNetwork, AcyclicNetwork

nn_seq = SequentialNetwork('seq', 'x')
nn_res = AcyclicNetwork('res', ['x'])

nn_seq.append_node(nodes.ReLUNode('R1', (1,)))
nn_seq.append_node(nodes.ReLUNode('R2', (1,)))
nn_seq.append_node(nodes.ReLUNode('R3', (1,)))

rl_1 = nodes.ReLUNode('R1', (1,))
rl_2 = nodes.ReLUNode('R2', (1,))
rl_3 = nodes.ReLUNode('R3', (1,))
rl_4 = nodes.ReLUNode('R4', (1,))
rl_5 = nodes.ReLUNode('R5', (1,))
rl_6 = nodes.ReLUNode('R6', (1,))

nn_res.add_node(rl_1)
nn_res.add_node(rl_2, [rl_1])
nn_res.add_node(rl_3, [rl_2])
nn_res.add_node(rl_4, [rl_2])
nn_res.add_node(rl_5, [rl_3])
nn_res.add_node(rl_6, [rl_4, rl_5])

print(nn_seq)
print(nn_res)

print(nn_seq.get_topological_order())
print(nn_res.get_topological_order())

print(nn_res.get_children(rl_2))
