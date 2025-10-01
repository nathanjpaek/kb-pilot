import torch


class TorchFloorDiv(torch.nn.Module):

    def __init__(self):
        super(TorchFloorDiv, self).__init__()

    def forward(self, x, y):
        return torch.floor_divide(x, y)


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
