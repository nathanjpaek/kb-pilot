import torch
from torch import Tensor
from torch import nn as nn
import torch.nn.functional as F


class MonteCarloDropout(nn.Dropout):
    """
    Defines Monte Carlo dropout Module as defined
    in the paper https://arxiv.org/pdf/1506.02142.pdf.
    In summary, This technique uses the regular dropout
    which can be interpreted as a Bayesian approximation of
    a well-known probabilistic model: the Gaussian process.
    We can treat the many different networks
    (with different neurons dropped out) as Monte Carlo samples
    from the space of all available models. This provides mathematical
    grounds to reason about the model’s uncertainty and, as it turns out,
    often improves its performance.
    """
    mc_dropout_enabled: 'bool' = False

    def train(self, mode: 'bool'=True):
        if mode:
            self.mc_dropout_enabled = True

    def forward(self, input: 'Tensor') ->Tensor:
        return F.dropout(input, self.p, self.mc_dropout_enabled, self.inplace)


class FeedForward(nn.Module):
    """
    ## FFN module

    source
    [FeedForward network](https://arxiv.org/abs/2002.05202)
    """

    def __init__(self, d_model: 'int', d_ff: 'int', dropout: 'float'=0.1,
        activation=nn.ReLU(), is_gated: 'bool'=False, bias1: 'bool'=True,
        bias2: 'bool'=True, bias_gate: 'bool'=True):
        """
        * `d_model` is the number of features in a token embedding
        * `d_ff` is the number of features in the hidden layer of the FFN
        * `dropout` is dropout probability for the hidden layer,
           compatible with Monte Carlo dropout at inference time
        * `is_gated` specifies whether the hidden layer is gated
        * `bias1` specified whether the first fully connected layer should have a learnable bias
        * `bias2` specified whether the second fully connected layer should have a learnable bias
        * `bias_gate` specified whether the fully connected layer for the gate should have a learnable bias
        """
        super().__init__()
        self.layer1 = nn.Linear(d_model, d_ff, bias=bias1)
        self.layer2 = nn.Linear(d_ff, d_model, bias=bias2)
        self.dropout = MonteCarloDropout(dropout)
        self.activation = activation
        self.is_gated = is_gated
        if is_gated:
            self.linear_v = nn.Linear(d_model, d_ff, bias=bias_gate)

    def forward(self, x: 'torch.Tensor'):
        g = self.activation(self.layer1(x))
        if self.is_gated:
            x = g * self.linear_v(x)
        else:
            x = g
        x = self.dropout(x)
        return self.layer2(x)


class Bilinear(nn.Module):

    def __init__(self, d_model: 'int', d_ff: 'int', dropout: 'float'=0.1):
        super().__init__()
        self.ffn = FeedForward(d_model, d_ff, dropout, nn.Identity(), True,
            False, False, False)

    def forward(self, x: 'torch.Tensor'):
        return self.ffn(x)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'d_model': 4, 'd_ff': 4}]
