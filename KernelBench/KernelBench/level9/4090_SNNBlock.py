from torch.nn import Module
import math
import torch
from torch.nn import SELU
from torch.nn import AlphaDropout
from torch.nn import Identity
from torch.nn import Parameter
from torch.nn.functional import conv2d


class SNNBlock(Module):
    """Block for a self-normalizing fully-connected layer.

    This block consists of:
        * AlphaDropout
        * Linear
        * SELU
    """

    def __init__(self, in_features: 'int', out_features: 'int', dropout:
        'float'=0.0, activation: 'bool'=True):
        """Initialize the layers.

        Args:
            in_features: The no. of input features
            out_features: The no. of output features
            dropout: The probability of dropping out the inputs
            activation: Whether to add the activation function
        """
        super().__init__()
        self.dropout = AlphaDropout(dropout)
        self.activation = SELU() if activation else Identity()
        stddev = math.sqrt(1 / in_features)
        weight = torch.randn(out_features, in_features, 1, 1) * stddev
        bias = torch.zeros(out_features)
        self.weight = Parameter(weight)
        self.bias = Parameter(bias)

    def forward(self, inputs: 'torch.Tensor') ->torch.Tensor:
        """Get the block's outputs."""
        outputs = self.dropout(inputs)
        outputs = conv2d(outputs, self.weight, self.bias)
        return self.activation(outputs)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_features': 4, 'out_features': 4}]
