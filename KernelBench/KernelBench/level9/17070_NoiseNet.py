import torch
import torch.nn.functional as F
from torch import nn


class NoiseNet(nn.Module):

    def __init__(self, channels=3, kernel_size=5):
        super(NoiseNet, self).__init__()
        self.kernel_size = kernel_size
        self.channels = channels
        to_pad = int((self.kernel_size - 1) / 2)
        self.padder = nn.ReflectionPad2d(to_pad).type(torch.FloatTensor)
        to_pad = 0
        self.convolver = nn.Conv2d(channels, channels, self.kernel_size, 1,
            padding=to_pad, bias=True).type(torch.FloatTensor)

    def forward(self, x):
        assert x.shape[1] == self.channels, (x.shape, self.channels)
        first = F.relu(self.convolver(self.padder(x)))
        second = F.relu(self.convolver(self.padder(first)))
        third = F.relu(self.convolver(self.padder(second)))
        assert x.shape == third.shape, (x.shape, third.shape)
        return third


def get_inputs():
    return [torch.rand([4, 3, 4, 4])]


def get_init_inputs():
    return [[], {}]
