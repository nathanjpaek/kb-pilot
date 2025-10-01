import torch


class AbsModule(torch.nn.Module):

    def __init__(self):
        super(AbsModule, self).__init__()

    def forward(self, x):
        return torch.abs(x)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
