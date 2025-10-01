import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.a1 = nn.Conv2d(19, 64, kernel_size=3, padding=1)
        self.a2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.a3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.b1 = nn.Conv2d(256, 256, kernel_size=3, padding=0)
        self.b2 = nn.Conv2d(256, 256, kernel_size=3, padding=0)
        self.b3 = nn.Conv2d(256, 256, kernel_size=4, padding=0)
        self.linear = nn.Linear(256, 7)

    def forward(self, x):
        x = F.relu(self.a1(x))
        x = F.relu(self.a2(x))
        x = F.relu(self.a3(x))
        x = F.relu(self.b1(x))
        x = F.relu(self.b2(x))
        x = F.relu(self.b3(x))
        x = x.view(-1, 256)
        x = self.linear(x)
        return x


def get_inputs():
    return [torch.rand([4, 19, 64, 64])]


def get_init_inputs():
    return [[], {}]
