import torch
import torch.utils.data
import torch.nn as nn
from torch.nn import functional as F


class Relation(nn.Module):

    def __init__(self, C, H, out_size):
        super(Relation, self).__init__()
        self.out_size = out_size
        self.M = torch.nn.Parameter(torch.randn(H, H, out_size))
        self.W = torch.nn.Parameter(torch.randn(C * out_size, C))
        self.b = torch.nn.Parameter(torch.randn(C))

    def forward(self, class_vector, query_encoder):
        mid_pro = []
        for slice in range(self.out_size):
            slice_inter = torch.mm(torch.mm(class_vector, self.M[:, :,
                slice]), query_encoder.transpose(1, 0))
            mid_pro.append(slice_inter)
        mid_pro = torch.cat(mid_pro, dim=0)
        V = F.relu(mid_pro.transpose(0, 1))
        probs = torch.sigmoid(torch.mm(V, self.W) + self.b)
        return probs


def get_inputs():
    return [torch.rand([4, 4]), torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'C': 4, 'H': 4, 'out_size': 4}]
