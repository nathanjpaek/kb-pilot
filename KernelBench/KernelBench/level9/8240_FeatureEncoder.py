import torch
from torch import nn
import torch.nn.functional as F


class FeatureEncoder(nn.Module):

    def __init__(self, video_dim, dim):
        super(FeatureEncoder, self).__init__()
        self.linear = nn.Linear(video_dim, dim)

    def forward(self, feature, h=None):
        feature = self.linear(feature)
        feature = F.leaky_relu(feature)
        return feature


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'video_dim': 4, 'dim': 4}]
