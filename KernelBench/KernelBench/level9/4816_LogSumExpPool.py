import torch
import torch.nn as nn


class LogSumExpPool(nn.Module):

    def __init__(self, gamma):
        super(LogSumExpPool, self).__init__()
        self.gamma = gamma

    def forward(self, feat_map):
        """
        Numerically stable implementation of the operation
        Arguments:
            feat_map(Tensor): tensor with shape (N, C, H, W)
            return(Tensor): tensor with shape (N, C, 1, 1)
        """
        _N, _C, H, W = feat_map.shape
        m, _ = torch.max(feat_map, dim=-1, keepdim=True)[0].max(dim=-2,
            keepdim=True)
        value0 = feat_map - m
        area = 1.0 / (H * W)
        g = self.gamma
        return m + 1 / g * torch.log(area * torch.sum(torch.exp(g * value0),
            dim=(-1, -2), keepdim=True))


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'gamma': 4}]
