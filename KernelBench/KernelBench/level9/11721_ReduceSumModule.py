import torch


class ReduceSumModule(torch.nn.Module):

    def __init__(self):
        super(ReduceSumModule, self).__init__()

    def forward(self, x):
        return torch.sum(x)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
