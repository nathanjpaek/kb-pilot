import torch
import torch.nn as nn


class GlobalMaxPool2d(nn.Module):

    def forward(self, inputs):
        return nn.functional.adaptive_max_pool2d(inputs, 1).view(inputs.
            size(0), -1)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
