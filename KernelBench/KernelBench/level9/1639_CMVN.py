import torch
import torch.nn as nn


class CMVN(nn.Module):
    __constants__ = ['mode', 'dim', 'eps']

    def __init__(self, mode='global', dim=2, eps=1e-10):
        super(CMVN, self).__init__()
        if mode != 'global':
            raise NotImplementedError(
                'Only support global mean variance normalization.')
        self.mode = mode
        self.dim = dim
        self.eps = eps

    def forward(self, x):
        if self.mode == 'global':
            return (x - x.mean(self.dim, keepdim=True)) / (self.eps + x.std
                (self.dim, keepdim=True))

    def extra_repr(self):
        return 'mode={}, dim={}, eps={}'.format(self.mode, self.dim, self.eps)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
