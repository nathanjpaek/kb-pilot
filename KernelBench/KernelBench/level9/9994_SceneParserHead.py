import torch
import torch.utils.data
from torch import nn


class SceneParserHead(nn.Module):

    def __init__(self, in_channels, num_classes):
        super(SceneParserHead, self).__init__()
        self.conv1x1 = nn.Conv2d(in_channels, 2048, 1, 1)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(2048, num_classes)

    def forward(self, x):
        x = self.conv1x1(x)
        x = self.avgpool(x)
        x = torch.flatten(x)
        x = self.fc(x)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 4, 'num_classes': 4}]
