import torch
import torch.nn as nn


class DynamicPreHead(nn.Module):

    def __init__(self, in_dim=3, embed_dim=100, kernel_size=1):
        super(DynamicPreHead, self).__init__()
        self.conv = nn.Conv2d(in_dim, embed_dim, kernel_size=kernel_size,
            stride=1, padding=int((kernel_size - 1) / 2))
        self.bn = nn.GroupNorm(int(embed_dim / 4), embed_dim)
        self.relu = nn.ReLU(True)
        nn.init.kaiming_normal_(self.conv.weight, mode='fan_out',
            nonlinearity='relu')

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


def get_inputs():
    return [torch.rand([4, 3, 64, 64])]


def get_init_inputs():
    return [[], {}]
