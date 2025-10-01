import torch
import torch.fx


class Norm(torch.nn.Module):

    def __init__(self):
        super(Norm, self).__init__()

    def forward(self, x):
        return torch.norm(x, 2, None, False)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
