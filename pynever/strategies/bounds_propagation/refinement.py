from enum import Enum

from pynever.strategies.verification.ssbp.constants import RefinementTarget


class NeuronSplit(Enum):
    Negative = 0
    Positive = 1


class RefiningBound(Enum):
    LowerBound = 1
    UpperBound = -1


class BoundsRefinement:

    @staticmethod
    def compute_refines_input_by(unstable, fixed_neurons, bounds, network):
        input_bounds = bounds['numeric_pre'][network.get_first_node().identifier]

        differences = list()
        for (layer_id, neuron_n) in unstable:
            negative_branch_input = BoundsRefinement.refine_input_bounds_after_split(
                bounds, network, RefinementTarget(layer_id, neuron_n), NeuronSplit.Negative, fixed_neurons)
            positive_branch_input = BoundsRefinement.refine_input_bounds_after_split(
                bounds, network, RefinementTarget(layer_id, neuron_n), NeuronSplit.Positive, fixed_neurons)

            if negative_branch_input is not None and positive_branch_input is not None:
                diff = \
                    ((negative_branch_input.get_lower() - input_bounds.get_lower()).sum() +
                     (input_bounds.get_upper() - negative_branch_input.get_upper()).sum() +
                     (input_bounds.get_upper() - positive_branch_input.get_upper()).sum() +
                     (positive_branch_input.get_lower() - input_bounds.get_lower()).sum())
            else:
                diff = 100

            if diff != 0:
                differences.append(((layer_id, neuron_n), diff))

        differences = sorted(differences, key=lambda x: x[1], reverse=True)
        return differences
