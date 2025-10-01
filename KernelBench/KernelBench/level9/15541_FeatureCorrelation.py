import torch
import torch.nn as nn


class FeatureCorrelation(nn.Module):

    def __init__(self, scale):
        super(FeatureCorrelation, self).__init__()
        self.scale = scale

    def forward(self, feature_A, feature_B):
        b, c, h, w = feature_A.size()
        feature_A = feature_A.transpose(2, 3).contiguous().view(b, c, h * w)
        feature_B = feature_B.view(b, c, h * w).transpose(1, 2)
        feature_mul = torch.bmm(feature_B, feature_A)
        correlation_tensor = self.scale * feature_mul.view(b, h, w, h * w
            ).transpose(2, 3).transpose(1, 2)
        return correlation_tensor


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'scale': 1.0}]
