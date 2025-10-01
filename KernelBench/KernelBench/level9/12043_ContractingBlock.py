import torch
from torch import nn


class ContractingBlock(nn.Module):

    def __init__(self, input_channels, use_bn=True, kernel_size=3,
        activation='relu'):
        super(ContractingBlock, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, input_channels * 2,
            kernel_size=kernel_size, padding=1, stride=2, padding_mode=
            'reflect')
        self.activation = nn.ReLU() if activation == 'relu' else nn.LeakyReLU(
            0.2)
        if use_bn:
            self.instancenorm = nn.InstanceNorm2d(input_channels * 2)
        self.use_bn = use_bn

    def forward(self, x):
        x = self.conv1(x)
        if self.use_bn:
            x = self.instancenorm(x)
        x = self.activation(x)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'input_channels': 4}]
