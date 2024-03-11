import numpy as np

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
        # Assignee: Pedro
        return 'Not verified'
    else:
        return 'Unsupported'


def split_star(star: Star, index: int, cur_bounds: AbstractBounds) -> list:
    # TODO split on target neuron (represent target neuron?)
    # Assignee: Stefano

    """
    For a star we only need the var_index to target a specific neuron.
    The index relative to this neuron is determined by the heuristic that
    also takes into account what layer the star comes from.

    When splitting I also need to update the bounds and return them
    """

    precision_guard = 10e-15
    lb = cur_bounds.get_lower()[index]
    ub = cur_bounds.get_upper()[index]

    mask = np.identity(star.center.shape[0])
    mask[index, index] = 0

    # Positive stable
    if lb >= precision_guard:
        return [(star, cur_bounds)]

    # Negative stable
    elif ub <= -precision_guard:
        new_c = np.matmul(mask, star.center)
        new_b = np.matmul(mask, star.basis_matrix)
        new_pred = star.predicate_matrix
        new_bias = star.predicate_bias
        return [(Star(new_pred, new_bias, new_c, new_b), cur_bounds)]

    # Unstable
    else:
        # Lower star
        lower_c = np.matmul(mask, star.center)
        lower_b = np.matmul(mask, star.basis_matrix)
        lower_pred = np.vstack((star.predicate_matrix, star.basis_matrix[index, :]))
        lower_bias = np.vstack((star.predicate_bias, -star.center[index]))

        # Upper star
        upper_c = star.center
        upper_b = star.basis_matrix
        upper_pred = np.vstack((star.predicate_matrix, -star.basis_matrix[index, :]))
        upper_bias = np.vstack((star.predicate_bias, star.center[index]))

        # TODO update bounds
        return [
            (Star(lower_pred, lower_bias, lower_c, lower_b), cur_bounds),
            (Star(upper_pred, upper_bias, upper_c, upper_b), cur_bounds)
        ]


def get_counterexample(star: Star) -> Tensor:
    return star.get_samples(num_samples=1)[0]


def get_target(star: Star, out_bounds: HyperRectangleBounds, nn: SequentialNetwork) -> 'TargetNeuron':
    # TODO select next neuron
    # Assignee: Stefano
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

    # Frontier is a stack of tuples (Star, AbstractBounds)
    frontier = [(in_star, out_bounds)]
    stop_flag = False

    while len(frontier) > 0 or not stop_flag:  # stop_flag for timeout interruption using signals (LATER)
        current_star, out_bounds = frontier.pop()
        result = check_intersection(out_bounds, prop)

        if result == 'Unknown':
            # How do I track partially split stars?
            target = params['heuristic'](current_star, out_bounds, nn)
            frontier.extend(
                split_star(current_star, target, out_bounds)
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
