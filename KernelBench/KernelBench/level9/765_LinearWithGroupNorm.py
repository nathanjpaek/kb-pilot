import torch
import torch.utils.data
from torch import nn
from math import gcd
import torch.cuda


class LinearWithGroupNorm(nn.Module):

    def __init__(self, n_in: 'int', n_out: 'int', num_groups: 'int'=32,
        activation: 'bool'=True) ->None:
        """
        Linear layer used in LaneGCN.
        :param n_in: Number of input channels.
        :param n_out: Number of output channels.
        :param num_groups: Number of groups for GroupNorm.
        :param activation: Boolean indicating whether to apply ReLU activation.
        """
        super().__init__()
        self.linear = nn.Linear(n_in, n_out, bias=False)
        self.norm = nn.GroupNorm(gcd(num_groups, n_out), n_out)
        self.relu = nn.ReLU(inplace=True)
        self.activation = activation

    def forward(self, x: 'torch.Tensor') ->torch.Tensor:
        """
        Apply linear layer to input tensor.
        :param x: Input tensor.
        :return: Output of linear layer.
        """
        out = self.linear(x)
        out = self.norm(out)
        if self.activation:
            out = self.relu(out)
        return out


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'n_in': 4, 'n_out': 4}]
