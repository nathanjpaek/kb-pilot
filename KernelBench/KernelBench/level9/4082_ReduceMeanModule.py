import torch


class ReduceMeanModule(torch.nn.Module):

    def __init__(self):
        super(ReduceMeanModule, self).__init__()

    def forward(self, x):
        return torch.mean(x)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
