import torch
import torch.nn as nn
import torch.nn.functional as F


class RC(nn.Module):
    """
    A wrapper class for ReflectionPad2d, Conv2d and an optional relu
    """

    def __init__(self, in_dim, out_dim, kernel_size=3, padding=1,
        activation_function=True):
        super().__init__()
        self.pad = nn.ReflectionPad2d((padding, padding, padding, padding))
        self.conv = nn.Conv2d(in_dim, out_dim, kernel_size)
        self.activation_function = activation_function

    def forward(self, x):
        x = self.pad(x)
        x = self.conv(x)
        return F.relu(x) if self.activation_function else x


class Decoder(nn.Module):

    def __init__(self):
        super(Decoder, self).__init__()
        self.rc1 = RC(512, 256, 3, 1)
        self.upsample1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.rc2 = RC(256, 256, 3, 1)
        self.rc3 = RC(256, 256, 3, 1)
        self.rc4 = RC(256, 256, 3, 1)
        self.rc5 = RC(256, 128, 3, 1)
        self.upsample2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.rc6 = RC(128, 128, 3, 1)
        self.rc7 = RC(128, 64, 3, 1)
        self.upsample3 = nn.Upsample(scale_factor=2, mode='nearest')
        self.rc8 = RC(64, 64, 3, 1)
        self.rc9 = RC(64, 3, 3, 1, False)

    def forward(self, x):
        x = self.rc1(x)
        x = self.upsample1(x)
        x = self.rc2(x)
        x = self.rc3(x)
        x = self.rc4(x)
        x = self.rc5(x)
        x = self.upsample2(x)
        x = self.rc6(x)
        x = self.rc7(x)
        x = self.upsample3(x)
        x = self.rc8(x)
        x = self.rc9(x)
        return x


def get_inputs():
    return [torch.rand([4, 512, 4, 4])]


def get_init_inputs():
    return [[], {}]
