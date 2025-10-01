import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPNet(nn.Module):

    def __init__(self):
        super(MLPNet, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 500)
        self.fc2 = nn.Linear(500, 256)
        self.fc3 = nn.Linear(256, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def name(self):
        return 'MLP'


def get_inputs():
    return [torch.rand([4, 784])]


def get_init_inputs():
    return [[], {}]
