import torch
import torch.nn as nn
import torch.utils.data


class VideoBoringModel(nn.Module):

    def __init__(self, in_channel):
        super().__init__()
        self.avg_pool3d = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Linear(in_channel, 1024)

    def forward(self, x):
        x = self.avg_pool3d(x).squeeze()
        x = self.fc(x)
        return x

    def output_size(self):
        return self.fc.in_features


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channel': 4}]
