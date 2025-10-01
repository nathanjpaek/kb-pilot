import torch
import torch.nn as nn


class LeakyClamp(nn.Module):

    def __init__(self, cap):
        super(LeakyClamp, self).__init__()
        self.cap = cap
        self.leakyrelu = nn.LeakyReLU(inplace=False)
        self.leakyrelu2 = nn.LeakyReLU(inplace=False)

    def forward(self, x):
        x = self.leakyrelu(x)
        x_ret = -self.leakyrelu2(-x + self.cap) + self.cap
        return x_ret


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'cap': 4}]
