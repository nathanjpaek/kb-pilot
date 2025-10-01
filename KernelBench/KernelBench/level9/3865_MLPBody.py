import torch
import torch.nn.functional as F
import torch.nn as nn


def layer_init(layer, w_scale=1.0):
    init_f = nn.init.orthogonal_
    init_f(layer.weight.data)
    layer.weight.data.mul_(w_scale)
    if layer.bias is not None:
        nn.init.constant_(layer.bias.data, 0)
    return layer


class MLPBody(nn.Module):

    def __init__(self, input_dim, feature_dim=512, hidden_dim=512):
        super().__init__()
        self.fc1 = layer_init(nn.Linear(input_dim, hidden_dim))
        self.fc2 = layer_init(nn.Linear(hidden_dim, feature_dim))
        self.feature_dim = feature_dim

    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x.view(x.size(0), -1))))


def get_inputs():
    return [torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'input_dim': 4}]
