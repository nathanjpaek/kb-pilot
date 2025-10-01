import torch
from torch import nn


class Out(nn.Module):

    def forward(self, out):
        out_std = torch.sqrt(out.var(0, unbiased=False) + 1e-08)
        mean_std = out_std.mean()
        mean_std = mean_std.expand(out.size(0), 1, 4, 4)
        out = torch.cat((out, mean_std), 1)
        return out


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
