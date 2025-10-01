import torch
import torch.nn as nn
import torch.nn.functional as F


def gem(x: 'torch.Tensor', p=3, eps=1e-06):
    return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(
        1.0 / p)


class GeM(nn.Module):

    def __init__(self, p=3, eps=1e-06):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return gem(x, p=self.p, eps=self.eps)

    def __repr__(self):
        return (self.__class__.__name__ +
            f'(p={self.p.data.tolist()[0]:.4f}, eps={self.eps})')


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
