import torch
from torch.nn import ReLU
from torch.nn import ReLU6
from torch.nn.functional import relu
from torch.nn.functional import relu6
from torch.nn import Conv2d
import torch.nn.functional


class ReLUBoundToPOTNet(torch.nn.Module):

    def __init__(self):
        super(ReLUBoundToPOTNet, self).__init__()
        self.conv1 = Conv2d(3, 3, kernel_size=1, stride=1)
        self.relu1 = ReLU6()
        self.conv2 = Conv2d(3, 3, kernel_size=1, stride=1)
        self.conv3 = Conv2d(3, 3, kernel_size=1, stride=1)
        self.conv4 = Conv2d(3, 3, kernel_size=1, stride=1)
        self.relu2 = ReLU()
        self.identity = torch.nn.Identity()

    def forward(self, inp):
        x = self.conv1(inp)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.identity(x)
        x = self.conv3(x)
        x = relu6(x)
        x = self.conv4(x)
        x = self.relu2(x)
        x = relu(x)
        return x


def get_inputs():
    return [torch.rand([4, 3, 64, 64])]


def get_init_inputs():
    return [[], {}]
