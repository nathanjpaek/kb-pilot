import torch
import torch.nn as nn
import torch.nn.functional as F


class CLeakyReLU(nn.LeakyReLU):

    def forward(self, xr, xi):
        return F.leaky_relu(xr, self.negative_slope, self.inplace
            ), F.leaky_relu(xi, self.negative_slope, self.inplace)


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
