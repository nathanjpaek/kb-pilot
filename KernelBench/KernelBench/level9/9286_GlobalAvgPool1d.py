import torch
import torch.nn as nn


class GlobalAvgPool1d(nn.Module):

    def __init__(self):
        """Global average pooling over the input's spatial dimensions"""
        super(GlobalAvgPool1d, self).__init__()

    def forward(self, inputs):
        return nn.functional.adaptive_avg_pool1d(inputs, 1).view(inputs.
            size(0), -1)


def get_inputs():
    return [torch.rand([4, 4])]


def get_init_inputs():
    return [[], {}]
