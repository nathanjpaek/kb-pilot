import torch
import torch.nn as nn


class PoseMap(nn.Module):

    def __init__(self):
        super(PoseMap, self).__init__()
        pass

    def forward(self, x):
        assert len(x.shape) == 4, 'The HeatMap shape should be BxCxHxW'
        res = x.sum(dim=1, keepdim=True)
        H = x.shape[2]
        W = x.shape[3]
        div = res.sum(dim=[2, 3], keepdim=True).repeat(1, 1, H, W)
        res = res / div
        return res


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
