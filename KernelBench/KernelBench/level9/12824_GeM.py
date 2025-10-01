import torch
from torch import nn
from torch.nn import functional as F
from torchvision.transforms import *


class GeM(nn.Module):

    def __init__(self, dim=2048, p=3, eps=1e-06):
        super(GeM, self).__init__()
        self.p = nn.Parameter(torch.ones(dim) * p, requires_grad=True)
        self.eps = eps
        self.dim = dim

    def forward(self, x):
        return self.gem(x, p=self.p, eps=self.eps)

    def gem(self, x, p=3, eps=1e-06):
        x = torch.transpose(x, 1, -1)
        x = x.clamp(min=eps).pow(p)
        x = torch.transpose(x, 1, -1)
        x = F.avg_pool2d(x, (x.size(-2), x.size(-1)))
        x = x.view(x.size(0), x.size(1))
        x = x.pow(1.0 / p)
        return x

    def __repr__(self):
        return self.__class__.__name__ + '(' + 'p=' + '{:.4f}'.format(self.
            p.data.tolist()[0]) + ', ' + 'eps=' + str(self.eps
            ) + ',' + 'dim=' + str(self.dim) + ')'


def get_inputs():
    return [torch.rand([4, 2048, 4])]


def get_init_inputs():
    return [[], {}]
