import numpy as np
import pandas as pd

from pynever import networks, nodes
from pynever.strategies import abstraction
from pynever.strategies.verification import NeVerProperty

if __name__ == '__main__':

    # Read weights from file (w25 = 0)
    df = pd.read_excel('data/SI_weights.xlsx')
    weights = df['linear_regression_weights'].to_numpy().reshape((1, 31))
    bias = np.array([-0.287451626])

    # Create FC layer
    nn = networks.SequentialNetwork('movendo_net', 'X')
    fc_1 = nodes.FullyConnectedNode('FC_1', (31,), 1, weights, bias)
    nn.add_node(fc_1)

    # Create logistic layer
    nn.add_node(nodes.SigmoidNode('SIG_1', (1,)))

    # Create abstract network
    abst_nn = abstraction.AbsSeqNetwork('movendo_abs_net')
    abst_nn.add_node(abstraction.AbsFullyConnectedNode('ABS_FC_1', fc_1))
    abst_nn.add_node(abstraction.AbsSigmoidNode('ABS_SIG_1', nodes.SigmoidNode('SIG', (1,)),
                                                B=3, M=0.42, approx_levels=5))

    # Create input star (center in normal values)
    norm_df = pd.read_excel('data/SI_info.xlsx', sheet_name='value_normality')
    range_df = pd.read_excel('data/SI_info.xlsx', sheet_name='value_range')

    center = norm_df['mean_normality'].to_numpy()
    lbs = range_df['value_range [min]'].to_numpy()
    ubs = range_df['value_range [max]'].to_numpy()

    noise = 0.05

    with open('prop.vnnlib', 'w') as vnn_prop:
        for i in range(31):
            vnn_prop.write(f'(declare-const X_{i} Real)\n')

        vnn_prop.write('\n(declare-const Y_0 Real)\n\n')

        for i in range(31):
            vnn_prop.write(f'(assert (<= X_{i} '
                           f'{center[i] + noise if center[i] + noise < ubs[i] else float(ubs[i])}))\n')
            vnn_prop.write(f'(assert (>= X_{i} '
                           f'{center[i] - noise if center[i] - noise > lbs[i] else float(lbs[i])}))\n')

        vnn_prop.write('\n(assert (>= Y_0 50.0))')

    to_verify = NeVerProperty()
    to_verify.from_smt_file('prop.vnnlib')
    abs_input = abstraction.StarSet({abstraction.Star(to_verify.in_coef_mat, to_verify.in_bias_mat)})

    for _, abs_node in abst_nn.nodes.items():
        abs_input = abs_node.forward(abs_input)

    for out_star in abs_input.stars:
        print(out_star)
