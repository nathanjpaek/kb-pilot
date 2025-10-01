import torch
import torch.nn.functional as F
import torch.nn as nn
from collections import OrderedDict


class PNet(nn.Module):

    def __init__(self):
        super(PNet, self).__init__()
        self.features = nn.Sequential(OrderedDict([('conv1', nn.Conv2d(3, 
            10, 3, 1)), ('prelu1', nn.PReLU(10)), ('pool1', nn.MaxPool2d(2,
            2, ceil_mode=True)), ('conv2', nn.Conv2d(10, 16, 3, 1)), (
            'prelu2', nn.PReLU(16)), ('conv3', nn.Conv2d(16, 32, 3, 1)), (
            'prelu3', nn.PReLU(32))]))
        self.conv4_1 = nn.Conv2d(32, 2, 1, 1)
        self.conv4_2 = nn.Conv2d(32, 4, 1, 1)

    def forward(self, x):
        """
        Arguments:
            x: a float tensor with shape [batch_size, 3, h, w].
        Returns:
            b: a float tensor with shape [batch_size, 4, h', w'].
            a: a float tensor with shape [batch_size, 2, h', w'].
        """
        x = self.features(x)
        a = self.conv4_1(x)
        b = self.conv4_2(x)
        a = F.softmax(a)
        return b, a


def get_inputs():
    return [torch.rand([4, 3, 64, 64])]


def get_init_inputs():
    return [[], {}]
