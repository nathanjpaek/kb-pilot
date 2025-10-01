import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data


class AttentiveTrans2d(nn.Module):

    def __init__(self, num_features, hidden_channels=32):
        super(AttentiveTrans2d, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.smooth_gamma = 1
        self.smooth_beta = 0
        self.matrix1 = nn.Parameter(torch.ones(num_features, hidden_channels))
        self.matrix2 = nn.Parameter(torch.ones(hidden_channels, num_features))
        self.sigmoid = nn.Sigmoid()
        self.conv1 = nn.Conv2d(2, 1, 3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(1, 1, 3, padding=1, bias=False)
        self.conv3 = nn.Conv2d(num_features, num_features, 1, bias=False)
        self.conv4 = nn.Conv2d(num_features, num_features, 1, bias=False)
        self.IN_norm = nn.InstanceNorm2d(num_features, affine=False,
            track_running_stats=False)

    def forward(self, feature):
        output = self.IN_norm(feature)
        feature_nc = self.avgpool(feature).view(feature.size()[0], feature.
            size()[1])
        channel_wise_response = self.sigmoid(feature_nc @ self.matrix1
            ) @ self.matrix2
        channel_wise_response = channel_wise_response.unsqueeze(-1).unsqueeze(
            -1).expand(output.size())
        avg_out = F.adaptive_avg_pool3d(feature, (1, feature.size()[2],
            feature.size()[3]))
        max_out = F.adaptive_max_pool3d(feature, (1, feature.size()[2],
            feature.size()[3]))
        avg_max_concat = torch.cat([avg_out, max_out], dim=1)
        spatial_wise_response = self.conv2(self.sigmoid(self.conv1(
            avg_max_concat))).expand(output.size())
        pixel_wise_response = channel_wise_response * spatial_wise_response
        importance_gamma = self.conv3(pixel_wise_response) + self.smooth_gamma
        importance_beta = self.conv4(pixel_wise_response) + self.smooth_beta
        out_in = output * importance_gamma + importance_beta
        return out_in


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'num_features': 4}]
