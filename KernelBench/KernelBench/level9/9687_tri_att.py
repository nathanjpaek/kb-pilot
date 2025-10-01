import torch
import torch.nn as nn


class tri_att(nn.Module):

    def __init__(self):
        super(tri_att, self).__init__()
        self.feature_norm = nn.Softmax(dim=2)
        self.bilinear_norm = nn.Softmax(dim=2)

    def forward(self, x):
        n = x.size(0)
        c = x.size(1)
        h = x.size(2)
        w = x.size(3)
        f = x.reshape(n, c, -1)
        f_norm = self.feature_norm(f * 2)
        bilinear = f_norm.bmm(f.transpose(1, 2))
        bilinear = self.bilinear_norm(bilinear)
        trilinear_atts = bilinear.bmm(f).view(n, c, h, w).detach()
        return trilinear_atts


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
