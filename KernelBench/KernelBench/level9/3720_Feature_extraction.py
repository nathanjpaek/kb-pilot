import torch
from torchvision import transforms as transforms
import torch.nn as nn


class Feature_extraction(nn.Module):

    def __init__(self, k, p):
        super(Feature_extraction, self).__init__()
        self.conv_1 = nn.Conv2d(3, 64, kernel_size=5, padding=2)
        self.conv_2 = nn.Conv2d(64, 64, kernel_size=k, padding=p)
        self.conv_3 = nn.Conv2d(64, 128, kernel_size=k, padding=p)
        self.conv_4 = nn.Conv2d(128, 128, kernel_size=k, padding=p)
        self.conv_5 = nn.Conv2d(128, 256, kernel_size=k, padding=p)
        self.conv_6 = nn.Conv2d(256, 256, kernel_size=k, padding=p)
        self.conv_7 = nn.Conv2d(256, 128, 1)
        self.conv_8 = nn.Conv2d(128, 22, 1)
        self.pool = nn.MaxPool2d(3, stride=2, padding=1)

    def forward(self, x):
        out1 = self.conv_1(x)
        out2 = self.conv_2(out1)
        pool1 = self.pool(out2)
        out3 = self.conv_3(pool1)
        out4 = self.conv_4(out3)
        pool2 = self.pool(out4)
        out5 = self.conv_5(pool2)
        out6 = self.conv_6(out5)
        out7 = self.conv_7(out6)
        out = self.conv_8(out7)
        return out


def get_inputs():
    return [torch.rand([4, 3, 64, 64])]


def get_init_inputs():
    return [[], {'k': 4, 'p': 4}]
