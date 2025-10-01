import torch
import torch.nn as nn
import torch.nn


def featureL2Norm(feature):
    epsilon = 1e-06
    norm = torch.pow(torch.sum(torch.pow(feature, 2), 1) + epsilon, 0.5
        ).unsqueeze(1).expand_as(feature)
    return torch.div(feature, norm)


class FeatureCorrelation(torch.nn.Module):

    def __init__(self, shape='3D', normalization=True):
        super(FeatureCorrelation, self).__init__()
        self.normalization = normalization
        self.shape = shape
        self.ReLU = nn.ReLU()

    def forward(self, feature_A, feature_B):
        if self.shape == '3D':
            b, c, h, w = feature_A.size()
            feature_A = feature_A.transpose(2, 3).contiguous().view(b, c, h * w
                )
            feature_B = feature_B.view(b, c, h * w).transpose(1, 2)
            feature_mul = torch.bmm(feature_B, feature_A)
            correlation_tensor = feature_mul.view(b, h, w, h * w).transpose(
                2, 3).transpose(1, 2)
        elif self.shape == '4D':
            b, c, hA, wA = feature_A.size()
            b, c, hB, wB = feature_B.size()
            feature_A = feature_A.view(b, c, hA * wA).transpose(1, 2)
            feature_B = feature_B.view(b, c, hB * wB)
            feature_mul = torch.bmm(feature_A, feature_B)
            correlation_tensor = feature_mul.view(b, hA, wA, hB, wB).unsqueeze(
                1)
        if self.normalization:
            correlation_tensor = featureL2Norm(self.ReLU(correlation_tensor))
        return correlation_tensor


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
