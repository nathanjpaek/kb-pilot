import torch
import torch.nn as nn


class TwoNet(nn.Module):

    def __init__(self, n_features, embedding_dim=256):
        super(TwoNet, self).__init__()
        self.a1 = nn.Linear(n_features, embedding_dim)
        self.a2 = nn.Linear(embedding_dim, 2)

    def forward(self, x):
        x = torch.relu(self.a1(x))
        return torch.sigmoid(self.a2(x))


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'n_features': 4}]
