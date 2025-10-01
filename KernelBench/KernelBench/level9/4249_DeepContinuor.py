import torch
import torch.nn as nn
import torch.nn.functional as F


class DeepContinuor(nn.Module):

    def __init__(self, x_dim, h_dim, y_dim):
        super().__init__()
        self.layer1 = nn.Linear(x_dim, h_dim)
        self.layer2 = nn.Linear(h_dim, h_dim)
        self.layer3 = nn.Linear(h_dim, h_dim)
        self.layer4 = nn.Linear(h_dim, y_dim)

    def forward(self, x):
        x = F.relu(self.layer1(x) + x)
        x = F.relu(self.layer2(x) + x)
        x = F.relu(self.layer3(x) + x)
        x = self.layer4(x)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'x_dim': 4, 'h_dim': 4, 'y_dim': 4}]
