import torch
import torch.nn as nn


class MakeFeatures(nn.Module):
    """ Returns features to be used by PairDrift. """

    def __init__(self, in_dim, out_dim):
        super(MakeFeatures, self).__init__()
        self.single = nn.Linear(in_dim, out_dim)
        self.pair = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        pairs = x[..., None, :, :] - x[..., :, None, :]
        return self.single(x), self.pair(pairs), pairs


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_dim': 4, 'out_dim': 4}]
