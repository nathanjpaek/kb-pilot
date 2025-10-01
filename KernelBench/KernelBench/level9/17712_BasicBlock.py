import torch
import torch.nn.functional as F
from torch import nn


class BasicBlock(nn.Module):

    def __init__(self, input_dim, width, block_depth):
        super(BasicBlock, self).__init__()
        self.block_depth = block_depth
        self.conv1 = nn.Conv2d(input_dim, width, kernel_size=3, padding=1)
        if block_depth > 1:
            self.conv2 = nn.Conv2d(width, width, kernel_size=3, padding=1)
        if block_depth > 2:
            self.conv3 = nn.Conv2d(width, width, kernel_size=3, padding=1)
        if block_depth > 3:
            self.conv4 = nn.Conv2d(width, width, kernel_size=3, padding=1)
        if block_depth > 4:
            raise BaseException('block_depth > 4 is not implemented.')

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out1 = out
        if self.block_depth > 1:
            out = F.relu(self.conv2(out))
        if self.block_depth > 2:
            out = F.relu(self.conv3(out))
        if self.block_depth > 3:
            out = F.relu(self.conv4(out))
        return out + out1


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'input_dim': 4, 'width': 4, 'block_depth': 1}]
