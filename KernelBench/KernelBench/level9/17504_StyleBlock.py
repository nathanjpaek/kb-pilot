import torch
import torch.nn as nn
import torch.fft


class AdaptiveInstanceNormalization(nn.Module):

    def and__init__(self):
        super(AdaptiveInstanceNormalization, self).__init__()

    def forward(self, x, mean, std):
        whitened_x = torch.nn.functional.instance_norm(x)
        return whitened_x * std + mean


class StyleBlock(nn.Module):

    def __init__(self, in_f, out_f):
        super().__init__()
        self.conv = nn.Conv2d(in_f, out_f, 3, 1, 1, padding_mode='circular')
        self.adain = AdaptiveInstanceNormalization()
        self.lrelu = nn.LeakyReLU()

    def forward(self, x, mean, var):
        x = self.conv(x)
        x = self.adain(x, mean, var)
        x = self.lrelu(x)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand(
        [4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_f': 4, 'out_f': 4}]
