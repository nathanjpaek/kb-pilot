import torch
import torch.nn as nn
import torch.nn.functional as functional


class Net(nn.Module):

    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(28 ** 2, 64)
        self.layer2 = nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(-1, 28 ** 2)
        x = self.layer1(x)
        x = functional.relu(x)
        x = self.layer2(x)
        return x


def get_inputs():
    return [torch.rand([4, 784])]


def get_init_inputs():
    return [[], {}]
