from torch.nn import Module
import torch
from torch import nn


class Pool(Module):
    """多尺度特征融合，借鉴Inception网络结构"""

    def __init__(self):
        super(Pool, self).__init__()
        self.max1 = nn.MaxPool2d(5, 1, 2)
        self.max2 = nn.MaxPool2d(9, 1, 4)
        self.max3 = nn.MaxPool2d(13, 1, 6)

    def forward(self, input_):
        return torch.cat((self.max1(input_), self.max2(input_), self.max3(
            input_), input_), dim=1)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
