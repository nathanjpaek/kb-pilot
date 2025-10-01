import torch
import torch.nn as nn


class global_avg_pool2d(nn.Module):

    def forward(self, x):
        _, _, h, w = x.shape
        return nn.AvgPool2d(kernel_size=(h, w))(x)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
