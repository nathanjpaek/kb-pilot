import torch


class Model(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x: 'torch.Tensor', y: 'torch.Tensor'):
        return x * y


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
