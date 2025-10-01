import torch
from torch import nn as nn


class FeatureNorm(nn.Module):

    def __init__(self, num_features, feature_index=1, rank=4, reduce_dims=(
        2, 3), eps=0.001, include_bias=True):
        super(FeatureNorm, self).__init__()
        self.shape = [1] * rank
        self.shape[feature_index] = num_features
        self.reduce_dims = reduce_dims
        self.scale = nn.Parameter(torch.ones(self.shape, requires_grad=True,
            dtype=torch.float))
        self.bias = nn.Parameter(torch.zeros(self.shape, requires_grad=True,
            dtype=torch.float)) if include_bias else nn.Parameter(torch.
            zeros(self.shape, requires_grad=False, dtype=torch.float))
        self.eps = eps

    def forward(self, features):
        f_std = torch.std(features, dim=self.reduce_dims, keepdim=True)
        f_mean = torch.mean(features, dim=self.reduce_dims, keepdim=True)
        return self.scale * ((features - f_mean) / (f_std + self.eps).sqrt()
            ) + self.bias


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'num_features': 4}]
