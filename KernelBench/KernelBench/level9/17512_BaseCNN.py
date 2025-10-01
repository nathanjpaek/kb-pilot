import torch
import torch.nn as nn


class BaseCNN(nn.Module):

    def __init__(self):
        super(BaseCNN, self).__init__()
        self.conv = nn.Conv1d(in_channels=1, out_channels=512, kernel_size=
            64, stride=32, padding=16)
        self.deconv = nn.ConvTranspose1d(in_channels=512, out_channels=1,
            kernel_size=64, stride=32, padding=16)

    def forward(self, x):
        x.size(0)
        x.size(2)
        ft_ly = self.conv(x)
        output = self.deconv(ft_ly)
        return output


def get_inputs():
    return [torch.rand([4, 1, 64])]


def get_init_inputs():
    return [[], {}]
