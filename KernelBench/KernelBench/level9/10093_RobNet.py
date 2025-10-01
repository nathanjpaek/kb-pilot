import torch
from torch import nn
import torch.nn.functional as F


class RobNet(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=2, dilation=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, stride=1)
        self.conv5 = nn.Conv2d(128, 128, kernel_size=2, stride=1)
        self.rotate = nn.Conv2d(128, 1, kernel_size=2, stride=1)
        self.bbox = nn.Conv2d(128, 4, kernel_size=2, stride=1)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        x = F.relu(self.conv1(x), inplace=True)
        x = F.relu(self.conv2(x), inplace=True)
        x = F.relu(self.conv3(x), inplace=True)
        x = F.relu(self.conv4(x), inplace=True)
        x = F.relu(self.conv5(x), inplace=True)
        rotate = self.rotate(x)
        rotate = self.sig(rotate)
        bbox = self.bbox(x)
        bbox = self.sig(bbox)
        out = torch.cat((bbox, rotate), dim=1)
        return out


def get_inputs():
    return [torch.rand([4, 1, 64, 64])]


def get_init_inputs():
    return [[], {}]
