import torch
import torch.nn as nn


class ConcatAvgMaxPooling(nn.Module):

    def __init__(self, kernel_size=12, stride=1):
        super(ConcatAvgMaxPooling, self).__init__()
        self.avgpool = nn.AvgPool2d(kernel_size, stride=1)
        self.maxpool = nn.MaxPool2d(kernel_size, stride=1)

    def forward(self, x):
        x = torch.cat((self.avgpool(x), self.maxpool(x)), axis=1)
        return x


def get_inputs():
    return [torch.rand([4, 4, 64, 64])]


def get_init_inputs():
    return [[], {}]
