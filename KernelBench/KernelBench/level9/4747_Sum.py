import torch
import numpy as np
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable as Variable


class Sum(nn.Module):

    def __init__(self, in_channels, in_features, out_channels, dropout=0.0):
        """
        Create a Sum layer.

        Args:
            in_channels (int): Number of output channels from the previous layer.
            in_features (int): Number of input features.
            out_channels (int): Multiplicity of a sum node for a given scope set.
            dropout (float, optional): Dropout percentage.
        """
        super(Sum, self).__init__()
        self.in_channels = in_channels
        self.in_features = in_features
        self.out_channels = out_channels
        self.dropout = dropout
        assert out_channels > 0, 'Number of output channels must be at least 1, but was %s.' % out_channels
        in_features = int(in_features)
        ws = torch.randn(in_features, in_channels, out_channels)
        self.sum_weights = nn.Parameter(ws)
        self._bernoulli_dist = torch.distributions.Bernoulli(probs=dropout)
        self.out_shape = f'(N, {in_features}, C_in)'

    def forward(self, x):
        """
        Sum layer foward pass.

        Args:
            x: Input of shape [batch, in_features, in_channels].

        Returns:
            torch.Tensor: Output of shape [batch, in_features, out_channels]
        """
        if self.dropout > 0.0:
            r = self._bernoulli_dist.sample(x.shape).type(torch.bool)
            x[r] = np.NINF
        x = x.unsqueeze(3) + F.log_softmax(self.sum_weights, dim=1)
        x = torch.logsumexp(x, dim=2)
        return x

    def __repr__(self):
        return (
            'Sum(in_channels={}, in_features={}, out_channels={}, dropout={}, out_shape={})'
            .format(self.in_channels, self.in_features, self.out_channels,
            self.dropout, self.out_shape))


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 4, 'in_features': 4, 'out_channels': 4}]
