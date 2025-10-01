import torch
import torch.nn as nn


class MeanPoolConv(nn.Module):

    def __init__(self, input_dim, output_dim, kernel_size=3, biases=True):
        super().__init__()
        self.conv = nn.Conv2d(input_dim, output_dim, kernel_size, stride=1,
            padding=kernel_size // 2, bias=biases)

    def forward(self, inputs):
        output = inputs
        output = sum([output[:, :, ::2, ::2], output[:, :, 1::2, ::2],
            output[:, :, ::2, 1::2], output[:, :, 1::2, 1::2]]) / 4.0
        return self.conv(output)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'input_dim': 4, 'output_dim': 4}]
