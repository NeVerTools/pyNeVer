import copy

from pynever.strategies.abstraction.bounds_propagation import ReLUStatus


class BoundsStats:
    """
    This class collects the statistics about stability and
    approximation obtained through Bounds Propagation

    Attributes
    ----------
    stability_info: dict[dict[str, list[int]] | dict[str, int]]
        Container for the statistics about neuron stability
    approximation_info: dict[tuple[str, int], float]
        Container for the statistics about the approximation area
    """

    def __init__(self):
        self.stability_info = {
            # These dictionaries are structured as
            # <layer_id: str> -> list[neuron: int]
            ReLUStatus.ACTIVE: dict(),
            ReLUStatus.INACTIVE: dict(),
            ReLUStatus.UNSTABLE: dict(),
            'stable_count': 0
        }

        # This dictionary is structured as
        # <(layer_id: str, neuron: int)> -> area: float
        self.approximation_info = dict()

    def count_unstable(self) -> int:
        """Procedure to count the unstable neurons"""
        return sum([len(v) for v in self.stability_info[ReLUStatus.UNSTABLE].values()])


class VerboseBounds:
    """
    This class is a complete collection of different bounds and information
    about them, used in the bounds propagation loop

    Attributes
    ----------
    identifiers: list[str]
        The list of identifiers of explored layers
    symbolic_bounds: dict[layer_id: str, bounds: SymbolicLinearBounds]
        The dictionary of the symbolic bounds for the network
    numeric_pre_bounds: dict[layer_id: str, bounds: HyperRectangleBounds]
        The dictionary of the numeric pre-activation bounds for the network
    numeric_post_bounds: dict[layer_id: str, bounds: HyperRectangleBounds]
        The dictionary of the numeric post-activation bounds for the network
    statistics: BoundsStats
        The data structure of stability and approximation statistics
    """

    def __init__(self,
                 identifiers: list = None,
                 symbolic: dict = None,
                 numeric_pre: dict = None,
                 numeric_post: dict = None):
        self.identifiers = copy.deepcopy(identifiers) if identifiers is not None else list()
        self.symbolic_bounds = copy.deepcopy(symbolic) if symbolic is not None else dict()
        self.numeric_pre_bounds = copy.deepcopy(numeric_pre) if numeric_pre is not None else dict()
        self.numeric_post_bounds = copy.deepcopy(numeric_post) if numeric_post is not None else dict()
        self.statistics = BoundsStats()
