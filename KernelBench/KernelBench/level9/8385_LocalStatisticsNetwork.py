import torch
import torch.nn as nn


class LocalStatisticsNetwork(nn.Module):

    def __init__(self, img_feature_channels: 'int'):
        """Local statistique nerwork

        Args:
            img_feature_channels (int): [Number of input channels]
        """
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=img_feature_channels,
            out_channels=512, kernel_size=1, stride=1)
        self.conv2 = nn.Conv2d(in_channels=512, out_channels=512,
            kernel_size=1, stride=1)
        self.conv3 = nn.Conv2d(in_channels=512, out_channels=1, kernel_size
            =1, stride=1)
        self.relu = nn.ReLU()

    def forward(self, concat_feature: 'torch.Tensor') ->torch.Tensor:
        x = self.conv1(concat_feature)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        local_statistics = self.conv3(x)
        return local_statistics


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'img_feature_channels': 4}]
