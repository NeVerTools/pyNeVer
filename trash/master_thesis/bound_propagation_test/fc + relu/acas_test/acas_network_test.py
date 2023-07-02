import pynever.utilities as utilities
import pynever.strategies.verification as ver
import pynever.nodes as nodes
import pynever.networks as networks
import time
from pynever.strategies import smt_reading, verification

if __name__ == '__main__':

    weights, biases, inputMeans, inputRanges, outputMean, outputRange = \
        utilities.parse_nnet("ACASXU_experimental_v2a_1_1.nnet")

    # Construction of our internal representation for the ACAS net.

    net = networks.SequentialNetwork("ACASXU_experimental_v2a_1_1", "X")

    for k in range(len(weights)):

        new_fc_node = nodes.FullyConnectedNode(f"FC_{k}", (weights[k].shape[1],), weights[k].shape[0], weights[k],
                                               biases[k], True)
        net.add_node(new_fc_node)

        if k < len(weights) - 1:
            new_relu_node = nodes.ReLUNode(f"ReLU_{k}", (weights[k].shape[0],))
            net.add_node(new_relu_node)

    path = "SMT_P3.smt2"
    parser = smt_reading.SmtPropertyParser(path, 'X', 'Y')
    prop = verification.NeVerProperty(*parser.parse_property())

    # heuristic, params and verifier
    heuristic = "best_n_neurons"
    params = [[1000] for _ in range(20)]
    verifier = ver.NeverVerification(heuristic, params)

    time_start = time.perf_counter()
    # Remember: we are verifying the negation of the original property
    safe = not verifier.verify(net, prop, 3,
                               "/Users/andrea/Documents/PYCHARM/pyNeVer/pynever/bound_propagation_test/fc + relu/acas_test/test_results/gimelli_bounds.csv",
                               "/Users/andrea/Documents/PYCHARM/pyNeVer/pynever/bound_propagation_test/fc + relu/acas_test/test_results/elena_bounds.csv")
    time_end = time.perf_counter()
