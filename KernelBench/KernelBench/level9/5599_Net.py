import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(2970, 1024)
        self.fc2 = nn.Linear(1024, 1)

    def forward(self, x, y=None):
        x = x.view(-1, 2970)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x


def get_inputs():
    return [torch.rand([4, 2970])]


def get_init_inputs():
    return [[], {}]
