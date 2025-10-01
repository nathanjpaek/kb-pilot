import torch
import torch.nn as nn
import torch.nn.functional as F


class encoderDepth(nn.Module):

    def __init__(self):
        super(encoderDepth, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=13, out_channels=64, kernel_size
            =4, stride=2, padding=1, bias=True)
        self.gn1 = nn.GroupNorm(4, 64)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size
            =3, stride=1, padding=1, bias=True)
        self.gn2 = nn.GroupNorm(4, 64)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=256,
            kernel_size=4, stride=2, padding=1, bias=True)
        self.gn3 = nn.GroupNorm(16, 256)
        self.conv4 = nn.Conv2d(in_channels=256, out_channels=256,
            kernel_size=3, stride=1, padding=1, bias=True)
        self.gn4 = nn.GroupNorm(16, 256)
        self.conv5 = nn.Conv2d(in_channels=256, out_channels=512,
            kernel_size=4, stride=2, padding=1, bias=True)
        self.gn5 = nn.GroupNorm(32, 512)
        self.conv6 = nn.Conv2d(in_channels=512, out_channels=512,
            kernel_size=3, stride=1, padding=1, bias=True)
        self.gn6 = nn.GroupNorm(32, 512)

    def forward(self, x):
        x1 = F.relu(self.gn1(self.conv1(x)), True)
        x2 = F.relu(self.gn2(self.conv2(x1)), True)
        x3 = F.relu(self.gn3(self.conv3(x2)), True)
        x4 = F.relu(self.gn4(self.conv4(x3)), True)
        x5 = F.relu(self.gn5(self.conv5(x4)), True)
        x = F.relu(self.gn6(self.conv6(x5)), True)
        return x


def get_inputs():
    return [torch.rand([4, 13, 64, 64])]


def get_init_inputs():
    return [[], {}]
