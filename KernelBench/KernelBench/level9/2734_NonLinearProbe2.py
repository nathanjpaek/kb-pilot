import torch
from torch import nn


class NonLinearProbe2(nn.Module):

    def __init__(self, input_dim, num_hidden=300, num_classes=255):
        super().__init__()
        self.linear1 = nn.Linear(in_features=input_dim, out_features=num_hidden
            )
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(in_features=num_hidden, out_features=
            num_classes)

    def forward(self, feature_vectors):
        x = self.linear1(feature_vectors)
        x = self.relu(x)
        return self.linear2(x)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'input_dim': 4}]
