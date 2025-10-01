import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init


class SVDBilinear(nn.Module):
    """
    my bilinear matmul but reducing parameter dimension using peusodu-SVD
    """

    def __init__(self, num_basis, in1_features, in2_features, out_features):
        super(SVDBilinear, self).__init__()
        self.num_basis = num_basis
        self.in1_features = in1_features
        self.in2_features = in2_features
        self.out_features = out_features
        self.left_singular = nn.Parameter(torch.Tensor(out_features,
            in1_features, num_basis))
        self.right_singular = nn.Parameter(torch.Tensor(out_features,
            num_basis, in2_features))
        self.diag = nn.Parameter(torch.Tensor(out_features, 1, num_basis))
        self.reset_parameter()

    def reset_parameter(self):
        init.xavier_uniform_(self.left_singular, gain=1.414)
        init.xavier_uniform_(self.right_singular, gain=1.414)
        init.normal_(self.diag, 0, 1 / math.sqrt(self.diag.size(-1)))

    def forward(self, in1, in2):
        us = self.left_singular * self.diag
        usv = torch.matmul(us, self.right_singular)
        return F.bilinear(in1, in2, weight=usv)

    def __repr__(self):
        return (
            'SVDBilinear Layer: in1_features={}, in2_features={}, out_features={}, num_basis={}'
            .format(self.in1_features, self.in2_features, self.out_features,
            self.num_basis))


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'num_basis': 4, 'in1_features': 4, 'in2_features': 4,
        'out_features': 4}]
