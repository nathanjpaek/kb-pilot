import torch
import torch.utils.data


class Abs(torch.nn.Module):

    def __init__(self):
        super(Abs, self).__init__()

    def forward(self, input):
        return torch.abs(input)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
