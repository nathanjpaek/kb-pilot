import torch


class Scale1Minus1(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = x / 254.0 * 2 - 1
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
