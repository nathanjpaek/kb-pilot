import torch
import torch.nn as nn
import torch.optim
import torch.utils.data
import torch.backends.cudnn


class UpsampleBlock(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(64, 256, 3, 1, 1)
        self.shuffle = nn.PixelShuffle(2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.shuffle(x)
        x = self.relu(x)
        return x


def get_inputs():
    return [torch.rand([4, 64, 64, 64])]


def get_init_inputs():
    return [[], {}]
