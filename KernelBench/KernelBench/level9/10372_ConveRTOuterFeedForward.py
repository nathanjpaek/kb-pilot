import torch
import torch.nn as nn
import torch.nn.functional as fnn
from torch.nn.modules.normalization import LayerNorm


class ConveRTOuterFeedForward(nn.Module):
    """Fully-Connected 3-layer Linear Model"""

    def __init__(self, input_hidden: 'int', intermediate_hidden: 'int',
        dropout_rate: 'float'=0.0):
        """
        :param input_hidden: first-hidden layer input embed-dim
        :type input_hidden: int
        :param intermediate_hidden: layer-(hidden)-layer middle point weight
        :type intermediate_hidden: int
        :param dropout_rate: dropout rate, defaults to None
        :type dropout_rate: float, optional
        """
        super().__init__()
        self.linear_1 = nn.Linear(input_hidden, intermediate_hidden)
        self.dropout = nn.Dropout(dropout_rate)
        self.linear_2 = nn.Linear(intermediate_hidden, intermediate_hidden)
        self.dropout = nn.Dropout(dropout_rate)
        self.linear_3 = nn.Linear(intermediate_hidden, intermediate_hidden)
        self.norm = LayerNorm(intermediate_hidden)

    def forward(self, x: 'torch.Tensor') ->torch.Tensor:
        """forward through fully-connected 3-layer

        :param x: fnn input
        :type x: torch.Tensor
        :return: return fnn output
        :rtype: torch.Tensor
        """
        x = self.linear_1(x)
        x = self.linear_2(self.dropout(x))
        x = self.linear_3(self.dropout(x))
        return fnn.gelu(self.norm(x))


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'input_hidden': 4, 'intermediate_hidden': 4}]
