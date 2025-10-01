import torch
import torch.nn as nn
from torch.nn import Module as Layer


class BiaffineAttention(Layer):
    """Implements a biaffine attention operator for binary relation classification."""

    def __init__(self, in_features, out_features):
        super(BiaffineAttention, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bilinear = nn.Bilinear(in_features, in_features, out_features,
            bias=False)
        self.linear = nn.Linear(2 * in_features, out_features)

    def forward(self, x_1, x_2):
        return self.bilinear(x_1, x_2) + self.linear(torch.cat((x_1, x_2),
            dim=-1))


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_features': 4, 'out_features': 4}]
