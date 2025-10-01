import torch
from torch import nn


class Model(nn.Module):

    def forward(self, img: 'torch.Tensor', scale: 'torch.Tensor', mean:
        'torch.Tensor'):
        return torch.div(torch.sub(img, mean), scale)


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand(
        [4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
