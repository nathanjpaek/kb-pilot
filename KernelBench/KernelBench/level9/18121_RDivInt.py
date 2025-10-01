import torch


class RDivInt(torch.nn.Module):

    def __init__(self):
        super(RDivInt, self).__init__()

    def forward(self, x):
        return 100 / x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
