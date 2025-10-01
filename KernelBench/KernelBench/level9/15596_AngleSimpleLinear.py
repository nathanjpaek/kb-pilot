import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data.distributed
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F


class AngleSimpleLinear(nn.Module):
    """Computes cos of angles between input vectors and weights vectors"""

    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        self.weight.data.normal_().renorm_(2, 1, 1e-05).mul_(100000.0)

    def forward(self, x):
        cos_theta = F.normalize(x.view(x.shape[0], -1), dim=1).mm(F.
            normalize(self.weight, p=2, dim=0))
        return cos_theta.clamp(-1, 1)

    def get_centers(self):
        return torch.t(self.weight)


def get_inputs():
    return [torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'in_features': 4, 'out_features': 4}]
