import torch
import numpy as np
import torch.nn as nn
import torch.nn.init
from math import log


def _mask_input(input, mask=None):
    if mask is not None:
        input = input * mask
        count = torch.sum(mask).data[0]
    else:
        count = np.prod(input.size(), dtype=np.float32).item()
    return input, count


class Log10Loss(nn.Module):

    def forward(self, input, target, mask=None):
        loss = torch.abs((torch.log(target) - torch.log(input)) / log(10))
        loss, count = _mask_input(loss, mask)
        return torch.sum(loss) / count


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
