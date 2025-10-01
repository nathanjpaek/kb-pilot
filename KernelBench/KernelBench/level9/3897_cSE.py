import torch
import torch.nn as nn


class cSE(nn.Module):

    def __init__(self, in_channels):
        super().__init__()
        reduced_filters = 1 if in_channels // 2 == 0 else in_channels // 2
        self.global_avg_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.pointwise_1 = nn.Conv2d(in_channels=in_channels, out_channels=
            reduced_filters, kernel_size=1)
        self.pointwise_2 = nn.Conv2d(in_channels=reduced_filters,
            out_channels=in_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU6()

    def forward(self, input_tensor):
        x = self.global_avg_pool(input_tensor)
        x = self.pointwise_1(x)
        x = self.relu(x)
        x = self.pointwise_2(x)
        x = self.sigmoid(x)
        x = torch.mul(input_tensor, x)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 4}]
