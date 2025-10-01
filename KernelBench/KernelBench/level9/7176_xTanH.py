import torch
import torch.nn


class xTanH(torch.nn.Module):

    def forward(self, x: 'torch.Tensor'):
        return x - torch.tanh(x)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
