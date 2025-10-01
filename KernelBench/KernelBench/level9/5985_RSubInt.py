import torch


class RSubInt(torch.nn.Module):

    def __init__(self):
        super(RSubInt, self).__init__()

    def forward(self, x):
        return 1 - x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
