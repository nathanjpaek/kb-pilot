import math
import torch
from torch.nn import functional as F
import torch.nn as nn
import torch.nn.init as init


class NormalizeLinear(nn.Module):

    def __init__(self, in_features, num_class):
        super(NormalizeLinear, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(num_class, in_features))
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, x):
        w = F.normalize(self.weight.float(), p=2, dim=1)
        return F.linear(x.float(), w)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_features': 4, 'num_class': 4}]
