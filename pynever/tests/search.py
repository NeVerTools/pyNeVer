import time

import pynever.strategies.bp.bounds_manager as bm
import pynever.strategies.search_functions as sf
from pynever.networks import SequentialNetwork
from pynever.strategies import conversion
from pynever.strategies.conversion import ONNXNetwork, ONNXConverter
from pynever.strategies.verification import NeVerProperty


def verify(prop: NeVerProperty, nn: SequentialNetwork, params: dict) -> list:
    in_star = prop.to_input_star()
    in_star.ref_layer = 0

    # The bounds here are a dict (key: layer.id)
    nn_bounds = sf.get_bounds(nn, prop, params['bounds'])

    # Frontier is a stack of tuples (Star, AbstractBounds)
    frontier = [(in_star, nn_bounds)]
    stop_flag = False

    # Init target refinement neuron (first index for the layer, second for the neuron)
    target = (0, 0)

    # Translate the network in a list
    net_list = bm.net2list(nn)

    while len(frontier) > 0 and not stop_flag:  # stop_flag for timeout interruption using signals (LATER)
        current_star, nn_bounds = frontier.pop()
        out_star = sf.abs_propagation(current_star, nn_bounds, target, net_list)
        intersects, unsafe_stars = sf.check_intersection(out_star, prop)

        # OK FINO QUI
        if intersects:
            if target[1] > current_star.center.shape[0]:
                if target[0] > len(net_list):
                    # Not verified
                    cex = sf.get_counterexample(out_star)
                    return ['Not verified', cex]
                else:
                    # Increment the layer
                    target = (target[0] + 1, 0)
            else:
                # Increment the neuron
                target = (target[0], target[1] + 1)

            # Unknown, target updated
            frontier.extend(
                sf.split_star(current_star, target[1], nn_bounds)
            )

    if stop_flag:
        return ['Unknown', 'parameters']
    else:
        return ['Verified']


if __name__ == '__main__':
    parameters = {
        'heuristic': sf.get_target,
        'bounds': 'symbolic'
    }

    start = time.perf_counter()
    network = conversion.load_network_path('../tests/data/acas.onnx')
    if isinstance(network, ONNXNetwork):
        network = ONNXConverter().to_neural_network(network)
    property = NeVerProperty()
    property.from_smt_file('../tests/data/acas.vnnlib', output_name='FC6')

    print(verify(property, network, parameters))

    # strategy = NeverVerification('overapprox', None)
    # print(strategy.verify(network, property))

    print(f'Elapsed time: {time.perf_counter() - start}')
