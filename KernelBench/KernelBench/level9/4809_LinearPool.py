import torch
import torch.nn as nn


class LinearPool(nn.Module):

    def __init__(self):
        super(LinearPool, self).__init__()

    def forward(self, feat_map):
        """
        Arguments:
            feat_map(Tensor): tensor with shape (N, C, H, W)
            return(Tensor): tensor with shape (N, C, 1, 1)
        """
        EPSILON = 1e-07
        _N, _C, _H, _W = feat_map.shape
        sum_input = torch.sum(feat_map, dim=(-1, -2), keepdim=True)
        sum_input += EPSILON
        linear_weight = feat_map / sum_input
        weighted_value = feat_map * linear_weight
        return torch.sum(weighted_value, dim=(-1, -2), keepdim=True)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
