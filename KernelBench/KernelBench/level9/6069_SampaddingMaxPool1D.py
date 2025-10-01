import torch
import torch.nn as nn


class SampaddingMaxPool1D(nn.Module):

    def __init__(self, pooling_size, stride):
        super(SampaddingMaxPool1D, self).__init__()
        self.pooling_size = pooling_size
        self.stride = stride
        self.padding = nn.ConstantPad1d((int((pooling_size - 1) / 2), int(
            pooling_size / 2)), 0)
        self.maxpool1d = nn.MaxPool1d(self.pooling_size, stride=self.stride)

    def forward(self, X):
        X = self.padding(X)
        X = self.maxpool1d(X)
        return X


def get_inputs():
    return [torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'pooling_size': 4, 'stride': 1}]
