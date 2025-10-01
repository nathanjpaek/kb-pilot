from torch.nn import Module
import torch
import torch.multiprocessing
import torch


class MaxPPVPool1d(Module):
    """Drop-in replacement for AdaptiveConcatPool1d - multiplies nf by 2"""

    def forward(self, x):
        _max = x.max(dim=-1).values
        _ppv = torch.gt(x, 0).sum(dim=-1).float() / x.shape[-1]
        return torch.cat((_max, _ppv), dim=-1).unsqueeze(2)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
