import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data


class Color_MNIST_CNN(nn.Module):

    def __init__(self):
        super(Color_MNIST_CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, 1, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(128, 128, 3, 1, padding=1)
        self.conv4 = nn.Conv2d(128, 128, 3, 1, padding=1)
        self.bn0 = nn.GroupNorm(8, 64)
        self.bn1 = nn.GroupNorm(8, 128)
        self.bn2 = nn.GroupNorm(8, 128)
        self.bn3 = nn.GroupNorm(8, 128)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.bn0(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.bn1(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.bn2(x)
        x = self.conv4(x)
        x = F.relu(x)
        x = self.bn3(x)
        x = self.avgpool(x)
        x = x.view(len(x), -1)
        return x


def get_inputs():
    return [torch.rand([4, 3, 64, 64])]


def get_init_inputs():
    return [[], {}]
