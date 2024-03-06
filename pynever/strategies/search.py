import numpy as np

from pynever.networks import SequentialNetwork
from pynever.strategies.abstraction import Star
from pynever.strategies.bp.bounds import HyperRectangleBounds
from pynever.strategies.verification import NeVerProperty
from pynever.tensor import Tensor


def check_intersection(out_bounds: HyperRectangleBounds, property: NeVerProperty) -> str:
    return 'Not verified'


def split_star(star: Star, neuron: 'TargetNeuron') -> list:
    return [(star, None)]


def get_counterexample(star: Star, property: NeVerProperty, nn: SequentialNetwork) -> Tensor:
    return np.zeros((2,))


def get_target(star: Star, out_bounds: HyperRectangleBounds, nn: SequentialNetwork) -> 'TargetNeuron':
    print('Hi')


def get_bounds(kind: str) -> HyperRectangleBounds:
    return HyperRectangleBounds([1, 0], [2, 2])


def verify(prop: NeVerProperty, nn: SequentialNetwork, params: dict) -> list:
    in_star = prop.to_input_star()
    # out_bounds = BoundsManager(nn, prop).compute_bounds(params['bounds'])
    out_bounds = get_bounds(params['bounds'])

    frontier = [(in_star, out_bounds)]
    stop_flag = False

    while len(frontier) > 0 or not stop_flag:
        current_star, out_bounds = frontier.pop()
        result = check_intersection(out_bounds, prop)

        if result == 'Unknown':
            target = params['heuristic'](current_star, out_bounds, nn)
            frontier.extend(
                split_star(current_star, target)
            )

        elif result == 'Not verified':
            cex = get_counterexample(current_star, prop, nn)
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

    network = SequentialNetwork('net', 'X')
    property = NeVerProperty()

    print(verify(property, network, parameters))
