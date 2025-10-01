import torch
import torch.nn as nn


class MRAE(nn.Module):

    def __init__(self):
        super(MRAE, self).__init__()

    def forward(self, output, target, mask=None):
        relative_diff = torch.abs(output - target) / (target + 1.0 / 65535.0)
        if mask is not None:
            relative_diff = mask * relative_diff
        return torch.mean(relative_diff)


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
