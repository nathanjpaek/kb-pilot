import torch
import numpy as np
import torch.nn as nn


class FeatureEmbeddingLayer(nn.Module):

    def __init__(self, dim_feature, dim_model):
        super(FeatureEmbeddingLayer, self).__init__()
        self.dim_model = dim_model
        self.embed = nn.Linear(dim_feature, dim_model)

    def forward(self, x):
        out = self.embed(x)
        out = out * np.sqrt(self.dim_model)
        return out


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'dim_feature': 4, 'dim_model': 4}]
