import torch
import torch.nn as nn


class GatedPooling(nn.Module):
    """
        Gated pooling as defined in https://arxiv.org/abs/1509.08985
        This implementation is the LR variant
    """

    def __init__(self, kernel_size, filter):
        super(GatedPooling, self).__init__()
        self.avgpool = nn.AvgPool2d(kernel_size)
        self.maxpool = nn.MaxPool2d(kernel_size)
        self.transform = nn.Conv2d(filter, filter, kernel_size=kernel_size,
            stride=kernel_size)

    def forward(self, x):
        alpha = torch.sigmoid(self.transform(x))
        return alpha * self.maxpool(x) + (1 - alpha) * self.avgpool(x)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'kernel_size': 4, 'filter': 4}]
