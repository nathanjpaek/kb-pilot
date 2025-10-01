import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.functional
from torch.nn import Parameter
from torch.nn.parameter import Parameter
import torch.nn.parallel
import torch.utils.data
import torch.optim
import torch.utils.data.distributed
import torch.autograd


class GeM(nn.Module):

    def __init__(self, p=3.0, eps=1e-06, freeze_p=True):
        super(GeM, self).__init__()
        self.p = p if freeze_p else Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return F.adaptive_avg_pool2d(x.clamp(min=self.eps).pow(self.p), (1, 1)
            ).pow(1.0 / self.p)

    def __repr__(self):
        if isinstance(self.p, float):
            p = self.p
        else:
            p = self.p.data.tolist()[0]
        return self.__class__.__name__ + '(' + 'p=' + '{:.4f}'.format(p
            ) + ', ' + 'eps=' + str(self.eps) + ')'


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
