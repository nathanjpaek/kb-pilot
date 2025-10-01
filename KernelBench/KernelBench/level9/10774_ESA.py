import torch
from torch import nn
import torch.nn.functional as F


class ESA(nn.Module):

    def __init__(self, channel=64, reduction=4, bias=True):
        super(ESA, self).__init__()
        self.r_nc = channel // reduction
        self.conv1 = nn.Conv2d(channel, self.r_nc, kernel_size=1)
        self.conv21 = nn.Conv2d(self.r_nc, self.r_nc, kernel_size=1)
        self.conv2 = nn.Conv2d(self.r_nc, self.r_nc, kernel_size=3, stride=
            2, padding=0)
        self.conv3 = nn.Conv2d(self.r_nc, self.r_nc, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(self.r_nc, self.r_nc, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(self.r_nc, self.r_nc, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(self.r_nc, channel, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = F.max_pool2d(self.conv2(x1), kernel_size=7, stride=3)
        x2 = self.relu(self.conv3(x2))
        x2 = self.relu(self.conv4(x2))
        x2 = F.interpolate(self.conv5(x2), (x.size(2), x.size(3)), mode=
            'bilinear', align_corners=False)
        x2 = self.conv6(x2 + self.conv21(x1))
        return x.mul(self.sigmoid(x2))


def get_inputs():
    return [torch.rand([4, 64, 64, 64])]


def get_init_inputs():
    return [[], {}]
