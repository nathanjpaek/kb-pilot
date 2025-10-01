import torch
import torch.nn as nn
import torch.nn.functional
import torch.nn.parallel
import torch.utils.data
import torch.optim
import torch.utils.data.distributed


class tofp16(nn.Module):
    """
    Utility module that implements::

        def forward(self, input):
            return input.half()
    """

    def __init__(self):
        super(tofp16, self).__init__()

    def forward(self, input):
        return input.half()


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
