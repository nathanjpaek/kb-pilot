import torch


class CeilModule(torch.nn.Module):

    def __init__(self):
        super(CeilModule, self).__init__()

    def forward(self, x):
        return torch.ceil(x)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
