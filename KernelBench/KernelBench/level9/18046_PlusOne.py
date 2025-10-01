import torch
import torch.optim


class PlusOne(torch.nn.Module):

    def __init__(self):
        super(PlusOne, self).__init__()

    def forward(self, x: 'torch.Tensor') ->torch.Tensor:
        return x + 1.0


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
