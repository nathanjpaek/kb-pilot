import torch


class FloorModule(torch.nn.Module):

    def __init__(self):
        super(FloorModule, self).__init__()

    def forward(self, x):
        return torch.floor(x)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
