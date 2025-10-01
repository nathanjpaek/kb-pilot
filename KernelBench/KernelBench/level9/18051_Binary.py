import torch
import torch.optim


class Binary(torch.nn.Module):

    def __init__(self):
        super(Binary, self).__init__()

    def forward(self, tensor: 'torch.Tensor') ->torch.Tensor:
        return (tensor != 0.0).bool()


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
