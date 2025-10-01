import torch
import torch.nn as nn
import torch.cuda


class Bottle(nn.Module):

    def forward(self, input):
        if len(input.size()) <= 2:
            return super(Bottle, self).forward(input)
        size = input.size()[:2]
        out = super(Bottle, self).forward(input.view(size[0] * size[1], -1))
        return out.contiguous().view(size[0], size[1], -1)


class BottleLinear(Bottle, nn.Linear):
    pass


class LayerNorm(nn.Module):
    """ Layer normalization module """

    def __init__(self, d_hid, eps=0.001):
        super(LayerNorm, self).__init__()
        self.eps = eps
        self.a_2 = nn.Parameter(torch.ones(d_hid), requires_grad=True)
        self.b_2 = nn.Parameter(torch.zeros(d_hid), requires_grad=True)

    def forward(self, z):
        if z.size(1) == 1:
            return z
        mu = torch.mean(z, dim=1)
        sigma = torch.std(z, dim=1)
        if mu.dim() == 1:
            mu = mu.unsqueeze(1)
            sigma = sigma.unsqueeze(1)
        ln_out = (z - mu.expand_as(z)) / (sigma.expand_as(z) + self.eps)
        ln_out = ln_out.mul(self.a_2.expand_as(ln_out)) + self.b_2.expand_as(
            ln_out)
        return ln_out


class BottleLayerNorm(Bottle, LayerNorm):
    pass


class PositionwiseFeedForward(nn.Module):
    """ A two-layer Feed-Forward-Network."""

    def __init__(self, size, hidden_size, dropout=0.1):
        """
        Args:
            size(int): the size of input for the first-layer of the FFN.
            hidden_size(int): the hidden layer size of the second-layer
                              of the FNN.
            droput(float): dropout probability(0-1.0).
        """
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = BottleLinear(size, hidden_size)
        self.w_2 = BottleLinear(hidden_size, size)
        self.layer_norm = BottleLayerNorm(size)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        output = self.dropout(self.w_2(self.relu(self.w_1(x))))
        return self.layer_norm(output + residual)


def get_inputs():
    return [torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'size': 4, 'hidden_size': 4}]
