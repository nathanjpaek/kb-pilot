import torch


class ReduceMinModule(torch.nn.Module):

    def __init__(self):
        super(ReduceMinModule, self).__init__()

    def forward(self, x):
        return torch.min(x)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
