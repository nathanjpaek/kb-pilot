import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data


class UnderfitNet(nn.Module):

    def __init__(self):
        super(UnderfitNet, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def get_inputs():
    return [torch.rand([4, 784])]


def get_init_inputs():
    return [[], {}]
