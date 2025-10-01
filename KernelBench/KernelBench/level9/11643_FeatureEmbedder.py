import torch
import numpy as np
import torch.nn as nn
from torch.utils import tensorboard as tensorboard


class FeatureEmbedder(nn.Module):

    def __init__(self, d_feat, d_model):
        super(FeatureEmbedder, self).__init__()
        self.d_model = d_model
        self.embedder = nn.Linear(d_feat, d_model)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.embedder(x)
        x = x * np.sqrt(self.d_model)
        x = self.activation(x)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'d_feat': 4, 'd_model': 4}]
