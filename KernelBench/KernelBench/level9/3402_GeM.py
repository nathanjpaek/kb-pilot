import torch
import torch.nn.functional as F
import torch.nn as nn


class GeM(nn.Module):

    def __init__(self, p=3, eps=1e-06):
        super(GeM, self).__init__()
        self.p = p
        self.eps = eps

    def forward(self, x):
        return self.gem(x, p=self.p, eps=self.eps)

    def gem(self, x, p=3, eps=1e-06):
        return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))
            ).pow(1.0 / p)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
