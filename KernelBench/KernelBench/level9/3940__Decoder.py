import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data


class _Decoder(nn.Module):

    def __init__(self, z_dim):
        super(_Decoder, self).__init__()
        self.fc1 = nn.Linear(z_dim, 600)
        self.fc2 = nn.Linear(600, 600)
        self.fc3 = nn.Linear(600, 784)

    def forward(self, z):
        h = F.relu(self.fc1(z))
        h = F.relu(self.fc2(h))
        h = torch.sigmoid(self.fc3(h))
        return h


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'z_dim': 4}]
