import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter


def linear(input, weight, bias=None):
    if input.dim() == 2 and bias is not None:
        ret = torch.addmm(bias, input, weight.t())
    else:
        output = input.matmul(weight.t())
        if bias is not None:
            output += bias
        ret = output
    return ret


class FeatureRegression(nn.Module):
    """Feature regression: sum(w[i] * x[i])"""

    def __init__(self, input_size, output_size=1):
        super(FeatureRegression, self).__init__()
        self.weight = Parameter(torch.Tensor(output_size, input_size))
        nn.init.xavier_normal_(self.weight)

    def forward(self, inputs):
        return linear(inputs, self.weight)


class TemporalDecayRegression(nn.Module):
    """Temporal decay regression exp(-relu(sum(w[i] * x[i])))"""

    def __init__(self, input_size, output_size=1, interactions=False):
        super(TemporalDecayRegression, self).__init__()
        self.interactions = interactions
        if interactions:
            self.linear = nn.Linear(input_size, output_size)
        else:
            self.weight = Parameter(torch.Tensor(output_size, input_size))
            nn.init.xavier_normal_(self.weight)

    def forward(self, inputs):
        if self.interactions:
            w = self.linear(inputs)
        else:
            w = linear(inputs, self.weight)
        gamma = torch.exp(-F.relu(w))
        return gamma


class FeatureEmbedding(nn.Module):
    """Regression layer with temporal decay."""

    def __init__(self, input_size, output_size=1, interactions=False):
        super(FeatureEmbedding, self).__init__()
        if interactions:
            self.feature_reg = nn.Linear(input_size, output_size)
        else:
            self.feature_reg = FeatureRegression(input_size, output_size)
        self.temporal_decay = TemporalDecayRegression(input_size,
            output_size, interactions=interactions)

    def forward(self, inputs, deltas):
        """input size: [batch_size,features] or [batch_size,timesteps,features]"""
        x = self.feature_reg(inputs)
        gamma = self.temporal_decay(deltas)
        xc = x * gamma
        return xc


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'input_size': 4}]
