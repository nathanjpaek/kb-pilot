import torch
from torch import nn
import torch.nn.functional as F
import torch.nn.parallel


class folder(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, feature_map):
        N, _, H, W = feature_map.size()
        feature_map = F.unfold(feature_map, kernel_size=3, padding=1)
        feature_map = feature_map.view(N, -1, H, W)
        return feature_map


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
