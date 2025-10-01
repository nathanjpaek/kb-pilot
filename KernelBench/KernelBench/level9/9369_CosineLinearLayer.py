import torch
from torch import Tensor
import torch.nn as nn
from torch.nn import Parameter


class CosineLinearLayer(nn.Module):

    def __init__(self, in_features: 'int', out_features: 'int') ->None:
        super(CosineLinearLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-05).mul_(100000.0)

    def forward(self, input: 'Tensor') ->Tensor:
        x = input
        w = self.weight
        ww = w.renorm(2, 1, 1e-05).mul(100000.0)
        xlen = x.pow(2).sum(1).pow(0.5)
        wlen = ww.pow(2).sum(0).pow(0.5)
        cos_theta = x.mm(ww)
        cos_theta = cos_theta / xlen.view(-1, 1) / wlen.view(1, -1)
        cos_theta = cos_theta.clamp(-1.0, 1.0)
        cos_theta = cos_theta * xlen.view(-1, 1)
        return cos_theta


def get_inputs():
    return [torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'in_features': 4, 'out_features': 4}]
