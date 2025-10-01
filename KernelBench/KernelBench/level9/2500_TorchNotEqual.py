import torch


class TorchNotEqual(torch.nn.Module):

    def __init__(self):
        super(TorchNotEqual, self).__init__()

    def forward(self, x, y):
        return torch.ne(x, y)


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
