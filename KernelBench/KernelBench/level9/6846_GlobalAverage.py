import torch
import torch.nn as nn


class GlobalAverage(nn.Module):

    def __init__(self, rank=2, keepdims=False):
        """
        :param rank: dimension of image
        :param keepdims: whether to preserve shape after averaging
        """
        super().__init__()
        self.axes = list(range(2, 2 + rank))
        self.keepdims = keepdims

    def forward(self, x):
        return torch.mean(x, dim=self.axes, keepdim=self.keepdims)

    def extra_repr(self):
        return f'axes={self.axes}, keepdims={self.keepdims}'


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
