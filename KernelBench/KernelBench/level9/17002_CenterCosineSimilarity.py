import torch
import torch.nn as nn


class CenterCosineSimilarity(nn.Module):

    def __init__(self, feat_dim, num_centers, eps=1e-08):
        super(CenterCosineSimilarity, self).__init__()
        self.centers = nn.Parameter(torch.randn(num_centers, feat_dim))
        self.eps = eps

    def forward(self, feat):
        norm_f = torch.norm(feat, p=2, dim=-1, keepdim=True)
        feat_normalized = torch.div(feat, norm_f)
        norm_c = torch.norm(self.centers, p=2, dim=-1, keepdim=True)
        center_normalized = torch.div(self.centers, norm_c)
        return torch.mm(feat_normalized, center_normalized.t())


def get_inputs():
    return [torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'feat_dim': 4, 'num_centers': 4}]
