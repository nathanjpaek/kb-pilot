import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed


class LayerNorm(nn.Module):

    def __init__(self, eps=0.0001):
        super(LayerNorm, self).__init__()
        self.eps = eps

    def forward(self, x):
        x_shape = x.shape
        x = x.reshape(x_shape[0], -1)
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True) + self.eps
        x = x / std - mean / std
        x = x.view(x_shape)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
