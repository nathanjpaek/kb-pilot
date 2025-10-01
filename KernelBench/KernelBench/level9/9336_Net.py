import torch
import torch.nn as nn


class Net(nn.Module):

    def __init__(self, input_d):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_d, int(input_d / 2))

    def forward(self, x):
        x = torch.sigmoid(self.fc1(x))
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'input_d': 4}]
