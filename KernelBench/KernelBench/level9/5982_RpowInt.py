import torch


class RpowInt(torch.nn.Module):

    def __init__(self):
        super(RpowInt, self).__init__()

    def forward(self, x):
        return 2 ** x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
