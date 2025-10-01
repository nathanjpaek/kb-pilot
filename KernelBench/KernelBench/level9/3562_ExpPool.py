import torch
from torch import nn


class ExpPool(nn.Module):

    def __init__(self):
        super(ExpPool, self).__init__()

    def forward(self, feat_map):
        """
        Numerically stable implementation of the operation
        Arguments:
            feat_map(Tensor): tensor with shape (N, C, H, W)
            return(Tensor): tensor with shape (N, C, 1, 1)
        """
        EPSILON = 1e-07
        _N, _C, _H, _W = feat_map.shape
        m, _ = torch.max(feat_map, dim=-1, keepdim=True)[0].max(dim=-2,
            keepdim=True)
        sum_exp = torch.sum(torch.exp(feat_map - m), dim=(-1, -2), keepdim=True
            )
        sum_exp += EPSILON
        exp_weight = torch.exp(feat_map - m) / sum_exp
        weighted_value = feat_map * exp_weight
        return torch.sum(weighted_value, dim=(-1, -2), keepdim=True)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
