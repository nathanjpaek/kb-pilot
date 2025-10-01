import torch
import torch.nn as nn


class TorchAdd(nn.Module):
    """TorchAdd Module.
    """

    def forward(self, input_list):
        return input_list[0] + input_list[1]


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
