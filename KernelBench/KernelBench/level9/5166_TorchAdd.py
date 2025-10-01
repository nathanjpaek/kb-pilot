import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data


class TorchAdd(nn.Module):
    """
    TorchAdd Module.
    """

    def forward(self, input_list):
        return input_list[0] + input_list[1]


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
