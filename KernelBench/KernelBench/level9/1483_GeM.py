import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from torch.nn.parameter import Parameter


def gem(x, p=3, eps=1e-06):
    return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(
        1.0 / p)


class GeM(nn.Module):

    def __init__(self, p=3.0, eps=1e-06, freeze_p=True):
        super(GeM, self).__init__()
        self.p = p if freeze_p else Parameter(torch.ones(1) * p)
        self.eps = eps
        self.freeze_p = freeze_p

    def forward(self, x):
        return gem(x, p=self.p, eps=self.eps)

    def __repr__(self):
        if isinstance(self.p, float):
            p = self.p
        else:
            p = self.p.data.tolist()[0]
        return self.__class__.__name__ + '(' + 'p=' + '{:.4f}'.format(p
            ) + ', ' + 'eps=' + str(self.eps) + ', ' + 'freeze_p=' + str(self
            .freeze_p) + ')'


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
