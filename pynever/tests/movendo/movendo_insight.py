import numpy as np
import pandas as pd

from pynever import nodes
from pynever.strategies import abstraction
from pynever.strategies.incomplete import sampling
from pynever.strategies.verification import NeVerProperty

def movendo_sig(x):
    return 1.0 / (1.0 * np.exp(-3 * (x - 0.42)))

if __name__ == '__main__':

    # Read data from files (w25 = 0)
    df = pd.read_excel('data/SI_weights.xlsx', sheet_name='RL_weights')
    norm_df = pd.read_excel('data/SI_info.xlsx', sheet_name='value_normality')
    range_df = pd.read_excel('data/SI_info.xlsx', sheet_name='value_range')

    bias = np.array([-0.287451626])
    weights = df['linear_regression_weights'].to_numpy().reshape((1, 31))
    center = norm_df['mean_normality'].to_numpy()
    lbs = range_df['value_range [min]'].to_numpy()
    ubs = range_df['value_range [max]'].to_numpy()
    noise = 0.01

    # Create directly the abstract network
    abst_nn = abstraction.AbsSeqNetwork('movendo_abs_net')
    abst_nn.add_node(abstraction.AbsFullyConnectedNode('ABS_FC_1',
                                                       nodes.FullyConnectedNode('FC_1', (31,),
                                                                                1, weights, bias)))
    abst_nn.add_node(abstraction.AbsSigmoidNode('ABS_SIG_1', nodes.SigmoidNode('SIG', (1,)),
                                                k=3, x0=0.42, approx_levels=1))

    # Create input star (center in normal values)
    # Write property for building Star (easier)
    with open('prop.vnnlib', 'w') as vnn_prop:
        for i in range(29):
            vnn_prop.write(f'(declare-const X_{i} Real)\n')

        vnn_prop.write('\n(declare-const Y_0 Real)\n\n')

        for i in range(29):
            # TODO use % noise for non-normalized inputs?
            vnn_prop.write(f'(assert (<= X_{i} '
                           f'{center[i] + noise if center[i] + noise < ubs[i] else float(ubs[i])}))\n')
            vnn_prop.write(f'(assert (>= X_{i} '
                           f'{center[i] - noise if center[i] - noise > lbs[i] else float(lbs[i])}))\n')

    movendo_prop = NeVerProperty()
    movendo_prop.from_smt_file('prop.vnnlib')

    # Sample the input and compute output
    samples = sampling.sample_property(movendo_prop, n_samples=10000)
    outputs = []

    for sample in samples:
        outputs.append(movendo_sig(np.sum(sample * weights[0, :29]) + bias))

    print(f'Normality: {movendo_sig(np.sum(center[:29] * weights[0, :29]) + bias)}')

    print(max(outputs))

    # abs_input = abstraction.StarSet({abstraction.Star(movendo_prop.in_coef_mat, movendo_prop.in_bias_mat)})
    #
    # for _, abs_node in abst_nn.nodes.items():
    #     abs_input = abs_node.forward(abs_input)
    #
    # lbs = []
    # ubs = []
    # i = 0
    # for star in abs_input.stars:
    #     lb, ub = star.get_bounds(0)
    #     lbs.append(lb)
    #     ubs.append(ub)
    #
    # min_lb = np.min(np.array(lbs))
    # max_ub = np.max(np.array(ubs))
    #
    # print(max_ub)
