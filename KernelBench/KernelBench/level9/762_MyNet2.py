import torch
from torch import nn
from torch.nn import functional as F


class MyNet2(nn.Module):
    """Very simple network made with two fully connected layers"""

    def __init__(self):
        super(MyNet2, self).__init__()
        self.fc1 = nn.Linear(28 * 50, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = x.view(-1, 28 * 50)
        x = F.dropout(x, 0.5)
        x = F.relu(self.fc1(x))
        x = F.sigmoid(self.fc2(x))
        return x


def get_inputs():
    return [torch.rand([4, 1400])]


def get_init_inputs():
    return [[], {}]
