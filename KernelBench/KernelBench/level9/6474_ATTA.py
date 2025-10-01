import torch
import torch.nn as nn


class ATTA(nn.Module):

    def __init__(self):
        super(ATTA, self).__init__()
        self.conv1 = nn.Conv2d(3, 3, 16, padding='same', groups=1, bias=False)
        self.lr = nn.LeakyReLU(0.2)
        self.conv2 = nn.Conv2d(3, 3, 3, padding='same', groups=1, bias=False)
        torch.nn.init.dirac_(self.conv1.weight, 1)
        torch.nn.init.dirac_(self.conv2.weight, 1)
        self.conv1.weight.data += torch.randn_like(self.conv1.weight.data
            ) * 0.01
        self.conv2.weight.data += torch.randn_like(self.conv2.weight.data
            ) * 0.01

    def forward(self, x):
        x2 = self.conv1(x)
        x3 = self.lr(x2)
        x4 = self.conv2(x3)
        return x4


def get_inputs():
    return [torch.rand([4, 3, 4, 4])]


def get_init_inputs():
    return [[], {}]
