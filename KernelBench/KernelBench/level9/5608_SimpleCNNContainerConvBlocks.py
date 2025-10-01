import torch
import torch.nn.functional as F
import torch.nn as nn


class SimpleCNNContainerConvBlocks(nn.Module):

    def __init__(self, input_channel, num_filters, kernel_size, output_dim=10):
        super(SimpleCNNContainerConvBlocks, self).__init__()
        """
        A testing cnn container, which allows initializing a CNN with given dims
        We use this one to estimate matched output of conv blocks

        num_filters (list) :: number of convolution filters
        hidden_dims (list) :: number of neurons in hidden layers

        Assumptions:
        i) we use only two conv layers and three hidden layers (including the output layer)
        ii) kernel size in the two conv layers are identical
        """
        self.conv1 = nn.Conv2d(input_channel, num_filters[0], kernel_size)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(num_filters[0], num_filters[1], kernel_size)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        return x


def get_inputs():
    return [torch.rand([4, 4, 64, 64])]


def get_init_inputs():
    return [[], {'input_channel': 4, 'num_filters': [4, 4], 'kernel_size': 4}]
