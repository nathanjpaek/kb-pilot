import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed


class AddFunction(nn.Module):

    def __init__(self):
        super(AddFunction, self).__init__()

    def forward(self, x, y):
        return x + y


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
