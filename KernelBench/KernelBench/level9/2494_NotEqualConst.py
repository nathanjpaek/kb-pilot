import torch


class NotEqualConst(torch.nn.Module):

    def __init__(self):
        super(NotEqualConst, self).__init__()

    def forward(self, x):
        return x != 13.62


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
