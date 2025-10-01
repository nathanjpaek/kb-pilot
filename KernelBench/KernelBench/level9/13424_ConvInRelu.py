import torch
import numpy as np
import torch.nn as nn


class ConvInRelu(nn.Module):

    def __init__(self, channels_in, channels_out, kernel_size, stride=1):
        super(ConvInRelu, self).__init__()
        self.n_params = 0
        self.channels = channels_out
        self.reflection_pad = nn.ReflectionPad2d(int(np.floor(kernel_size / 2))
            )
        self.conv = nn.Conv2d(channels_in, channels_out, kernel_size,
            stride, padding=0)
        self.instancenorm = nn.InstanceNorm2d(channels_out)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x = self.reflection_pad(x)
        x = self.conv(x)
        x = self.instancenorm(x)
        x = self.relu(x)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'channels_in': 4, 'channels_out': 4, 'kernel_size': 4}]
