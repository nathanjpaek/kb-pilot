import torch
from torch import nn


class LinearNormalGamma(nn.Module):

    def __init__(self, in_chanels, out_channels):
        super().__init__()
        self.linear = nn.Linear(in_chanels, out_channels * 4)

    def evidence(self, x):
        return torch.log(torch.exp(x) + 1)

    def forward(self, x):
        pred = self.linear(x).view(x.shape[0], -1, 4)
        mu, logv, logalpha, logbeta = [w.squeeze(-1) for w in torch.split(
            pred, 1, dim=-1)]
        return mu, self.evidence(logv), self.evidence(logalpha
            ) + 1, self.evidence(logbeta)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_chanels': 4, 'out_channels': 4}]
