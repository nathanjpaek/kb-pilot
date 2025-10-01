import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter


class CosineLinear(nn.Module):

    def __init__(self, in_features, out_features):
        super(CosineLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, input):
        x = F.normalize(input, dim=-1)
        w = F.normalize(self.weight, dim=0)
        cos_theta = x.mm(w)
        return w, cos_theta


def get_inputs():
    return [torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'in_features': 4, 'out_features': 4}]
