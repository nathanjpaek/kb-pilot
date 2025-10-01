import torch
import torch.utils.data
import torch
from torch import nn


def resize(x1, x2, largest=True):
    if largest:
        if x1.size()[2:] > x2.size()[2:]:
            x2 = nn.Upsample(size=x1.size()[2:], mode='bilinear')(x2)
        elif x1.size()[2:] < x2.size()[2:]:
            x1 = nn.Upsample(size=x2.size()[2:], mode='bilinear')(x1)
        return x1, x2
    else:
        raise NotImplementedError


class ParamSum(nn.Module):

    def __init__(self, C):
        super(ParamSum, self).__init__()
        self.a = nn.Parameter(torch.ones(C))
        self.b = nn.Parameter(torch.ones(C))

    def forward(self, x, y):
        bsize = x.size(0)
        x, y = resize(x, y)
        return self.a.expand(bsize, -1)[:, :, None, None] * x + self.b.expand(
            bsize, -1)[:, :, None, None] * y


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'C': 4}]
