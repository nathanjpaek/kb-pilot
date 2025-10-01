import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data


class FCNet(nn.Module):

    def __init__(self):
        super(FCNet, self).__init__()
        self.fc1 = nn.Linear(3 * 28 * 28, 128)
        self.fc2 = nn.Linear(128, 5)

    def forward(self, x):
        x = x.view(-1, 3 * 28 * 28)
        x = torch.sigmoid(self.fc1(x))
        x = self.fc2(x)
        return x


def get_inputs():
    return [torch.rand([4, 2352])]


def get_init_inputs():
    return [[], {}]
