import torch
import torch.nn as nn


class crop(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        N, C, H, W = x.shape
        x = x[0:N, 0:C, 0:H - 1, 0:W]
        return x


class shift(nn.Module):

    def __init__(self):
        super().__init__()
        self.shift_down = nn.ZeroPad2d((0, 0, 1, 0))
        self.crop = crop()

    def forward(self, x):
        x = self.shift_down(x)
        x = self.crop(x)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
