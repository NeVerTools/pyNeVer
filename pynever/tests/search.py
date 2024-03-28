import logging
import time

import pynever.strategies.bp.bounds_manager as bm
import pynever.strategies.search_functions as sf
from pynever import nodes
from pynever.networks import SequentialNetwork
from pynever.strategies import conversion
from pynever.strategies.conversion import ONNXNetwork, ONNXConverter
from pynever.strategies.verification import NeVerProperty

logger_stream = logging.getLogger()
logger_stream.addHandler(logging.StreamHandler())
logger_stream.setLevel(logging.INFO)


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

    # Retrieve the last ReLU layer index
    last_relu_idx = 0
    for layer in net_list[::-1]:
        if isinstance(layer, nodes.ReLUNode):
            last_relu_idx = net_list.index(layer)
            break

    while len(frontier) > 0 and not stop_flag:  # stop_flag for timeout interruption using signals (LATER)
        logger_stream.info(len(frontier))
        current_star, nn_bounds = frontier.pop()

        out_star = sf.abs_propagation(current_star, nn_bounds, target, net_list)
        intersects, unsafe_stars = sf.check_intersection(out_star, prop)

        if intersects:
            # If new target is None there is no more refinement to do
            target, current_star = params['refinement'](current_star, target, net_list, last_relu_idx)

            if target is None:
                # Not verified
                cex = out_star.get_samples(num_samples=1)[0]
                return ['Not verified', cex]

            else:
                # Unknown, target updated
                frontier.extend(
                    sf.split_star(current_star, target, net_list, nn_bounds)
                )

    if stop_flag:
        return ['Unknown', 'parameters']
    else:
        return ['Verified']


if __name__ == '__main__':
    parameters = {
        'refinement': sf.get_target_sequential,
        'bounds': 'symbolic'
    }

    start = time.perf_counter()
    network = conversion.load_network_path('data/Networks/dubinsrejoin.onnx')

    if isinstance(network, ONNXNetwork):
        network = ONNXConverter().to_neural_network(network)

        if isinstance(network, SequentialNetwork):
            ver_property = NeVerProperty()
            ver_property.from_smt_file('data/Properties/dubinsrejoin_case_unsafe_52.vnnlib', output_name='Y')

            result = verify(ver_property, network, parameters)
            print(result)
            print(f'Elapsed time: {time.perf_counter() - start}')
