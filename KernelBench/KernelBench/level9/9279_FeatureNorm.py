import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data


class FeatureNorm(nn.Module):

    def __init__(self, eps=1e-06):
        super(FeatureNorm, self).__init__()
        self.eps = eps

    def forward(self, feature):
        norm_feat = torch.sum(torch.pow(feature, 2), 1) + self.eps
        norm_feat = torch.pow(norm_feat, 0.5).unsqueeze(1)
        norm_feat = norm_feat.expand_as(feature)
        norm_feat = torch.div(feature, norm_feat)
        return norm_feat


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
