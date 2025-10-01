import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data


class _Classifier(nn.Module):

    def __init__(self, z_c_dim):
        super(_Classifier, self).__init__()
        self.fc1 = nn.Linear(z_c_dim, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, z_c):
        h = F.relu(self.fc1(z_c))
        h = self.fc2(h)
        return h


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'z_c_dim': 4}]
