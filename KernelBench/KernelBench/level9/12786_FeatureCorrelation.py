import torch
from torch import nn


class FeatureCorrelation(nn.Module):

    def __init__(self):
        super(FeatureCorrelation, self).__init__()

    def forward(self, featureA, featureB):
        b, c, h, w = featureA.size()
        featureA = featureA.permute(0, 3, 2, 1).reshape(b, w * h, c)
        featureB = featureB.reshape(b, c, h * w)
        corr = torch.bmm(featureA, featureB).reshape(b, w * h, h, w)
        return corr


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
