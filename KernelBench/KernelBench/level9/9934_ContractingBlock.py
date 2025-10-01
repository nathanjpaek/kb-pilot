import torch
import torch.nn as nn


class ContractingBlock(nn.Module):

    def __init__(self, input_channel):
        super(ContractingBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=input_channel, out_channels=
            input_channel * 2, kernel_size=(3, 3))
        self.conv2 = nn.Conv2d(input_channel * 2, input_channel * 2,
            kernel_size=(3, 3))
        self.maxpool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.activation(self.conv1(x))
        x = self.activation(self.conv2(x))
        return self.maxpool(x)


def get_inputs():
    return [torch.rand([4, 4, 64, 64])]


def get_init_inputs():
    return [[], {'input_channel': 4}]
