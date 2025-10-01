import torch
import torch.nn as nn


class BaseEncoder(nn.Module):

    def __init__(self):
        super(BaseEncoder, self).__init__()
        self._input_dimensions = 0

    @property
    def input_dimensions(self):
        return self._input_dimensions


class MeanEncoder(BaseEncoder):

    def __init__(self):
        super(MeanEncoder, self).__init__()
        self._input_dimensions = 4 * 300

    def forward(self, x, x_len):
        out = torch.div(torch.sum(x, 0), x_len.float().view(-1, 1))
        return out


def get_inputs():
    return [torch.rand([4, 4]), torch.rand([4, 4])]


def get_init_inputs():
    return [[], {}]
