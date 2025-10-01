import torch


class TorchMod(torch.nn.Module):

    def __init__(self):
        super(TorchMod, self).__init__()

    def forward(self, x, y):
        return torch.fmod(x, y)


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
