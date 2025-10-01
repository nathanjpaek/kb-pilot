import torch
import torch.nn as nn


class CombineFeatures(nn.Module):
    """ Returns layer to be used by PairDrift. """

    def __init__(self, in_dim, out_dim, zero_init=False):
        super(CombineFeatures, self).__init__()
        self.single = nn.Linear(in_dim, out_dim)
        self.pair = nn.Linear(in_dim, out_dim)
        if zero_init:
            self.single.weight.data = torch.zeros(out_dim, in_dim)
            self.single.bias.data = torch.zeros(out_dim)
            self.pair.weight.data = torch.zeros(out_dim, in_dim)
            self.pair.bias.data = torch.zeros(out_dim)

    def forward(self, s, p):
        return self.single(s) + self.pair(p).sum(dim=-3)


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_dim': 4, 'out_dim': 4}]
