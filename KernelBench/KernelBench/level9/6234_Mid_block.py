import torch
import torch.nn as nn
import torch.utils.data


class Mid_block(nn.Module):

    def __init__(self, chanIn, chanOut, ks=3, stride=1):
        super().__init__()
        self.conv1 = nn.Conv3d(chanIn, chanOut, ks, padding=1)
        self.conv2 = nn.Conv3d(chanOut, chanOut, ks, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'chanIn': 4, 'chanOut': 4}]
