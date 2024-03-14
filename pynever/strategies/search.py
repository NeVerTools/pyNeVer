import search_functions as sf
from pynever.networks import SequentialNetwork
from pynever.strategies import conversion
from pynever.strategies.conversion import ONNXNetwork, ONNXConverter
from pynever.strategies.verification import NeVerProperty


def verify(prop: NeVerProperty, nn: SequentialNetwork, params: dict) -> list:
    in_star = prop.to_input_star()
    out_bounds = sf.get_bounds(params['bounds'], nn, prop)

    # Frontier is a stack of tuples (Star, AbstractBounds)
    frontier = [(in_star, out_bounds)]
    stop_flag = False
    complete = False

    while len(frontier) > 0 or not stop_flag:  # stop_flag for timeout interruption using signals (LATER)
        current_star, out_bounds = frontier.pop()
        out_star = sf.abs_propagation(current_star, out_bounds, nn)
        intersects, unsafe_stars = sf.check_intersection(out_star, prop)

        if intersects:
            if not complete:  # Unknown
                # How do I track partially split stars?
                target = params['heuristic'](out_star, out_bounds, nn)
                frontier.extend(
                    sf.split_star(out_star, target, out_bounds)
                )

            else:  # Not verified
                cex = sf.get_counterexample(out_star)
                return ['Not verified', cex]

    if stop_flag:
        return ['Unknown', 'parameters']
    else:
        return ['Verified']


if __name__ == '__main__':
    parameters = {
        'heuristic': sf.get_target,
        'bounds': 'symbolic'
    }

    network = conversion.load_network_path('../tests/data/acas.onnx')
    if isinstance(network, ONNXNetwork):
        network = ONNXConverter().to_neural_network(network)
    property = NeVerProperty()
    property.from_smt_file('../tests/data/acas.vnnlib', output_name='FC6')

    print(verify(property, network, parameters))
