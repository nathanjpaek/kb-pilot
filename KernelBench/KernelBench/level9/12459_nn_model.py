import torch
import torch.nn as nn
import torch.nn.functional as F


class nn_model(nn.Module):

    def __init__(self, feature_dim, num_classes):
        super(nn_model, self).__init__()
        self.l1 = nn.Linear(feature_dim, 1024)
        self.l2 = nn.Linear(1024, 1024)
        self.l3 = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = self.l3(x)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'feature_dim': 4, 'num_classes': 4}]
