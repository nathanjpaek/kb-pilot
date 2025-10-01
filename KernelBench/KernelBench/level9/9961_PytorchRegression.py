import torch
import torch.nn as nn
import torch.nn.functional as F


class PytorchRegression(nn.Module):

    def __init__(self, num_features):
        super(PytorchRegression, self).__init__()
        self.layer_1 = nn.Linear(num_features, 128)
        self.layer_out = nn.Linear(128, 1)

    def forward(self, x):
        x = F.dropout(F.relu(self.layer_1(x)))
        x = self.layer_out(x)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'num_features': 4}]
