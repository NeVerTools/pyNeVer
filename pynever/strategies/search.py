from pynever.networks import SequentialNetwork
from pynever.strategies import conversion
from pynever.strategies.abstraction import Star
from pynever.strategies.bp.bounds import HyperRectangleBounds, AbstractBounds, SymbolicLinearBounds
from pynever.strategies.bp.bounds_manager import BoundsManager
from pynever.strategies.conversion import ONNXNetwork, ONNXConverter
from pynever.strategies.verification import NeVerProperty
from pynever.tensor import Tensor


def check_intersection(out_bounds: AbstractBounds, property: NeVerProperty) -> str:
    if isinstance(out_bounds, SymbolicLinearBounds):
        # TODO intersect out_bounds with output property
        return 'Not verified'
    else:
        return 'Unsupported'


def split_star(star: Star, neuron: 'TargetNeuron') -> list:
    # TODO split on target neuron (represent target neuron?)
    return [(star, None)]


def get_counterexample(star: Star) -> Tensor:
    return star.get_samples(num_samples=1)[0]


def get_target(star: Star, out_bounds: HyperRectangleBounds, nn: SequentialNetwork) -> 'TargetNeuron':
    print('Hi')


def get_bounds(kind: str, nn: SequentialNetwork, prop: NeVerProperty) -> AbstractBounds:
    if kind == 'symbolic':
        return BoundsManager(nn, prop).compute_bounds()[0]
    elif kind == 'autoLirpa':
        # return something...
        pass


def verify(prop: NeVerProperty, nn: SequentialNetwork, params: dict) -> list:
    in_star = prop.to_input_star()
    out_bounds = get_bounds(params['bounds'], nn, prop)

    frontier = [(in_star, out_bounds)]
    stop_flag = False

    while len(frontier) > 0 or not stop_flag:  # stop_flag for timeout interruption using signals (LATER)
        current_star, out_bounds = frontier.pop()
        result = check_intersection(out_bounds, prop)

        if result == 'Unknown':
            # How do I track partially split stars?
            target = params['heuristic'](current_star, out_bounds, nn)
            frontier.extend(
                split_star(current_star, target)
            )

        elif result == 'Not verified':
            cex = get_counterexample(current_star)
            return ['Not verified', cex]

    if stop_flag:
        return ['Unknown', 'parameters']
    else:
        return ['Verified']


if __name__ == '__main__':
    parameters = {
        'heuristic': get_target,
        'bounds': 'symbolic'
    }

    network = conversion.load_network_path('../tests/data/acas.onnx')
    if isinstance(network, ONNXNetwork):
        network = ONNXConverter().to_neural_network(network)
    property = NeVerProperty()
    property.from_smt_file('../tests/data/acas.vnnlib', output_name='FC6')

    print(verify(property, network, parameters))
