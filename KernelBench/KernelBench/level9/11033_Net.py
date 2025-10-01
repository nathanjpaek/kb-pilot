from torch.nn import Module
import torch
from torch.nn import Conv2d
from torch.nn import Dropout2d
from torch.nn import Linear
from torch.nn.functional import relu
from torch.nn.functional import max_pool2d
from torch.nn.functional import log_softmax
from torch import flatten


class Net(Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = Conv2d(in_channels=1, out_channels=32, kernel_size=3,
            stride=1)
        self.conv2 = Conv2d(in_channels=32, out_channels=64, kernel_size=3,
            stride=1)
        self.dropout1 = Dropout2d(p=0.5)
        self.fc1 = Linear(in_features=12544, out_features=128)
        self.fc2 = Linear(in_features=128, out_features=40)

    def forward(self, x):
        x = self.conv1(x)
        x = relu(x)
        x = max_pool2d(x, 2)
        x = self.conv2(x)
        x = relu(x)
        x = max_pool2d(x, 2)
        x = flatten(x, 1)
        x = self.fc1(x)
        x = relu(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        out = log_softmax(x, dim=1)
        return out


def get_inputs():
    return [torch.rand([4, 1, 64, 64])]


def get_init_inputs():
    return [[], {}]
