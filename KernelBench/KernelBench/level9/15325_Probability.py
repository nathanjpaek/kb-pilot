import torch
import torch.nn as nn


class Probability(nn.Module):
    """A layer that predicts the probabilities
    """

    def __init__(self, n_primitives, input_channels, make_dense=False):
        super(Probability, self).__init__()
        self._n_primitives = n_primitives
        self._make_dense = make_dense
        if self._make_dense:
            self._fc = nn.Conv3d(input_channels, input_channels, 1)
            self._nonlin = nn.LeakyReLU(0.2, True)
        self._probability_layer = nn.Conv3d(input_channels, self.
            _n_primitives, 1)

    def forward(self, X):
        if self._make_dense:
            X = self._nonlin(self._fc(X))
        probs = torch.sigmoid(self._probability_layer(X)).view(-1, self.
            _n_primitives)
        return probs


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'n_primitives': 4, 'input_channels': 4}]
