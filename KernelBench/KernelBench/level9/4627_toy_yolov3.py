import torch
import torch.nn as nn
import torch.nn.functional as F


class toy_yolov3(nn.Module):

    def __init__(self):
        super(toy_yolov3, self).__init__()
        self.conv1 = nn.Conv2d(3, 128, kernel_size=3, stride=2, padding=1)
        self.conv2_1 = nn.Conv2d(128, 128, kernel_size=1, stride=1, padding=0)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(256, 255, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(255, 255, kernel_size=3, stride=2, padding=1)
        self.conv6 = nn.Conv2d(255, 255, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        identity = x
        out = self.conv2_1(x)
        out = self.conv2_2(out)
        out += identity
        out = self.conv3(out)
        out = F.interpolate(out, scale_factor=2)
        out = torch.cat((out, identity), dim=1)
        out1 = self.conv4(out)
        out2 = self.conv5(out1)
        out3 = self.conv6(out2)
        return out3


def get_inputs():
    return [torch.rand([4, 3, 64, 64])]


def get_init_inputs():
    return [[], {}]
