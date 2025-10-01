import torch
import torch.nn as nn
import torch.nn.functional as F


class FCTestNN(nn.Module):

    def __init__(self, class_size):
        super(FCTestNN, self).__init__()
        self.name = 'FCTestNN'
        self.fc1 = nn.Linear(3 * 224 * 224, 256)
        self.fc2 = nn.Linear(256, class_size)

    def forward(self, x):
        x = x.view(-1, 3 * 224 * 224)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = x.squeeze(1)
        return x


def get_inputs():
    return [torch.rand([4, 150528])]


def get_init_inputs():
    return [[], {'class_size': 4}]
