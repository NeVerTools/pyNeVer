class RefinementTarget:
    """
    This class represents the refinement target for the verification.

    """

    def __init__(self, layer: int, neuron: int):
        self.layer_idx = layer
        self.neuron_idx = neuron

    def __repr__(self):
        return f'({self.layer_idx}, {self.neuron_idx})'

    def to_pair(self):
        return self.layer_idx, self.neuron_idx
