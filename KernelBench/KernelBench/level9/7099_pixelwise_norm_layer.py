import torch
import torch.nn as nn
import torch.utils.data


class pixelwise_norm_layer(nn.Module):

    def __init__(self):
        super(pixelwise_norm_layer, self).__init__()
        self.eps = 1e-08

    def forward(self, x):
        return x / (torch.mean(x ** 2, dim=1, keepdim=True) + self.eps) ** 0.5


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
