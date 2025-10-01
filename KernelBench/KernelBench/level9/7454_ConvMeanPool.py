import torch
import torch.nn as nn


class ConvMeanPool(nn.Module):

    def __init__(self, input_dim, output_dim, kernel_size=3, biases=True,
        adjust_padding=False):
        super().__init__()
        if not adjust_padding:
            conv = nn.Conv2d(input_dim, output_dim, kernel_size, stride=1,
                padding=kernel_size // 2, bias=biases)
            self.conv = conv
        else:
            conv = nn.Conv2d(input_dim, output_dim, kernel_size, stride=1,
                padding=kernel_size // 2, bias=biases)
            self.conv = nn.Sequential(nn.ZeroPad2d((1, 0, 1, 0)), conv)

    def forward(self, inputs):
        output = self.conv(inputs)
        output = sum([output[:, :, ::2, ::2], output[:, :, 1::2, ::2],
            output[:, :, ::2, 1::2], output[:, :, 1::2, 1::2]]) / 4.0
        return output


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'input_dim': 4, 'output_dim': 4}]
