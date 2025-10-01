import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):

    def __init__(self, input_channel, output_channel, upsample=True):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(input_channel, output_channel, kernel_size=3,
            padding=0)
        self.conv2 = nn.Conv2d(output_channel, output_channel, kernel_size=
            3, padding=0)
        self.conv_shortcut = nn.Conv2d(input_channel, output_channel,
            kernel_size=1, bias=False)
        self.relu = nn.LeakyReLU(0.2)
        self.norm = nn.InstanceNorm2d(output_channel)
        self.upsample = upsample
        self.reflecPad1 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.reflecPad2 = nn.ReflectionPad2d((1, 1, 1, 1))

    def forward(self, x):
        if self.upsample:
            x = F.interpolate(x, mode='bilinear', scale_factor=2)
        x_s = self.conv_shortcut(x)
        x = self.conv1(self.reflecPad1(x))
        x = self.relu(x)
        x = self.norm(x)
        x = self.conv2(self.reflecPad2(x))
        x = self.relu(x)
        x = self.norm(x)
        return x_s + x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'input_channel': 4, 'output_channel': 4}]
