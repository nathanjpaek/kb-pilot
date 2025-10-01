import torch
from torch import nn


class NonLinearProbe1(nn.Module):

    def __init__(self, input_dim, num_classes=255):
        super().__init__()
        self.linear = nn.Linear(in_features=input_dim, out_features=num_classes
            )
        self.relu = nn.ReLU()

    def forward(self, feature_vectors):
        return self.relu(self.linear(feature_vectors))


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'input_dim': 4}]
