import torch
import torch._utils


class RAddFloat(torch.nn.Module):

    def __init__(self):
        super(RAddFloat, self).__init__()

    def forward(self, x):
        y = 1.0 + x
        y = y + y + 1
        y = y + y + 1
        x = y + x
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
