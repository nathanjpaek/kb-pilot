import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureNet(nn.Module):

    def __init__(self, state_dim, feature_dim):
        super(FeatureNet, self).__init__()
        self.l1 = nn.Linear(state_dim, 300)
        self.l2 = nn.Linear(300, feature_dim)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'state_dim': 4, 'feature_dim': 4}]
