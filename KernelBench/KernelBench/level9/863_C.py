import torch
import torch.nn as nn


class C(nn.Module):

    def __init__(self, input_channel, output_channel, kernel_size, stride,
        padding, activation=None):
        """
        At the final layer, a 3x3 convolution is used to map each 64-component feature vector to the desired
        number of classes.

        :param input_channel: input channel size
        :param output_channel: output channel size
        """
        super(C, self).__init__()
        if activation == 'sigmoid':
            self.layer = nn.Sequential([nn.Conv2d(input_channel,
                output_channel, kernel_size=kernel_size, stride=stride,
                padding=padding), nn.Sigmoid()])
        elif activation == 'tanh':
            self.layer = nn.Sequential([nn.Conv2d(input_channel,
                output_channel, kernel_size=kernel_size, stride=stride,
                padding=padding), nn.Tanh()])
        else:
            self.layer = nn.Conv2d(input_channel, output_channel,
                kernel_size=kernel_size, stride=stride, padding=padding)

    def forward(self, x):
        return self.layer(x)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'input_channel': 4, 'output_channel': 4, 'kernel_size': 4,
        'stride': 1, 'padding': 4}]
