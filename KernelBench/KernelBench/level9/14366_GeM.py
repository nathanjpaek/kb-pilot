import torch
import torch.nn as nn


class GeM(nn.Module):

    def __init__(self, p=3, eps=1e-06):
        super(GeM, self).__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return nn.functional.avg_pool2d(x.clamp(min=self.eps).pow(self.p),
            (x.size(-2), x.size(-1))).pow(1.0 / self.p)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
