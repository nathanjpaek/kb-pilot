import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class CosMargin(nn.Module):

    def __init__(self, in_size, out_size, s=None, m=0.0):
        super(CosMargin, self).__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.W = nn.Parameter(torch.randn(out_size, in_size), requires_grad
            =True)
        self.W.data.uniform_(-1, 1).renorm_(2, 1, 1e-05).mul_(100000.0)
        self.s = nn.Parameter(torch.randn(1), requires_grad=True
            ) if s is None else s
        self.m = m

    def forward(self, x, label=None):
        cosine = F.linear(F.normalize(x), F.normalize(self.W))
        if label is not None and math.fabs(self.m) > 1e-06:
            one_hot = torch.zeros_like(cosine)
            one_hot.scatter_(1, label.view(-1, 1), 1.0)
            output = (cosine - one_hot * self.m) * self.s
        else:
            output = cosine * self.s
        return output

    def __repr__(self):
        return (self.__class__.__name__ +
            '(in_size={}, out_size={}, s={}, m={})'.format(self.in_size,
            self.out_size, 'learn' if isinstance(self.s, nn.Parameter) else
            self.s, 'learn' if isinstance(self.m, nn.Parameter) else self.m))


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_size': 4, 'out_size': 4}]
