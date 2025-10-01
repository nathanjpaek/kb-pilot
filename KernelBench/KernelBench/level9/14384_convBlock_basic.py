import torch
from torch import nn


class convBlock_basic(nn.Module):

    def __init__(self, inChannel, outChannel, kernel, stride, pad,
        use_batchnorm=False):
        super(convBlock_basic, self).__init__()
        self.use_batchnorm = use_batchnorm
        self.conv = nn.Conv2d(inChannel, outChannel, kernel, stride=stride,
            padding=pad)
        if self.use_batchnorm:
            self.bn = nn.BatchNorm2d(outChannel)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        if self.use_batchnorm:
            x = self.bn(x)
        x = self.relu(x)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'inChannel': 4, 'outChannel': 4, 'kernel': 4, 'stride': 1,
        'pad': 4}]
