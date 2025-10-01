import torch
import torch.nn as nn


class FPNHead(nn.Module):

    def __init__(self, num_in, num_mid, num_out):
        super().__init__()
        self.block0 = nn.Conv2d(num_in, num_mid, kernel_size=3, padding=1,
            bias=False)
        self.block1 = nn.Conv2d(num_mid, num_out, kernel_size=3, padding=1,
            bias=False)

    def forward(self, x):
        x = nn.functional.relu(self.block0(x), inplace=True)
        x = nn.functional.relu(self.block1(x), inplace=True)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'num_in': 4, 'num_mid': 4, 'num_out': 4}]
