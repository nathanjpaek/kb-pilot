import torch
import torch.nn as nn


class LogLog(nn.Module):

    def forward(self, x):
        return 1.0 - torch.exp(-torch.exp(x))


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
