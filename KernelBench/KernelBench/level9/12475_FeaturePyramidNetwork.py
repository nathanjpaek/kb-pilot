import torch
import torch.nn as nn


class FeaturePyramidNetwork(nn.Module):

    def __init__(self, C3_feature, C4_feature, C5_feature, feature_size=256):
        super(FeaturePyramidNetwork, self).__init__()
        self.P5_1 = nn.Conv2d(C5_feature, feature_size, kernel_size=1,
            stride=1, padding=0)
        self.P5_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        self.P5_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3,
            stride=1, padding=1)
        self.P4_1 = nn.Conv2d(C4_feature, feature_size, kernel_size=1,
            stride=1, padding=0)
        self.P4_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        self.P4_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3,
            stride=1, padding=1)
        self.P3_1 = nn.Conv2d(C3_feature, feature_size, kernel_size=1,
            stride=1, padding=0)
        self.P3_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3,
            stride=1, padding=1)

    def forward(self, C3, C4, C5):
        P5_x = self.P5_1(C5)
        P5_upsampled_x = self.P5_upsampled(P5_x)
        P5_x = self.P5_2(P5_x)
        P4_x = self.P4_1(C4)
        P4_x = P5_upsampled_x + P4_x
        P4_upsampled_x = self.P4_upsampled(P4_x)
        P4_x = self.P4_2(P4_x)
        P3_x = self.P3_1(C3)
        P3_x = P3_x + P4_upsampled_x
        P3_x = self.P3_2(P3_x)
        return P3_x


def get_inputs():
    return [torch.rand([4, 4, 16, 16]), torch.rand([4, 4, 8, 8]), torch.
        rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'C3_feature': 4, 'C4_feature': 4, 'C5_feature': 4}]
