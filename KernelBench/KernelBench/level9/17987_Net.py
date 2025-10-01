import torch
import torch.nn as nn


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 3, kernel_size=7, stride=1, bias=False,
            padding=3)
        self.conv2 = nn.Conv2d(3, 3, kernel_size=7, stride=1, bias=False,
            padding=3)
        self.conv3 = nn.Conv2d(3, 3, kernel_size=7, stride=1, bias=False,
            padding=3)
        self.conv4 = nn.Conv2d(3, 16, kernel_size=7, stride=2, bias=False,
            padding=3)
        self.conv5 = nn.Conv2d(16, 32, kernel_size=7, stride=2, bias=False,
            padding=3)
        self.conv6 = nn.Conv2d(32, 64, kernel_size=7, stride=1, bias=False,
            padding=3)
        self.conv7 = nn.Conv2d(64, 64, kernel_size=3, stride=1, bias=False,
            padding=1)
        self.conv8 = nn.Conv2d(64, 64, kernel_size=3, stride=2, bias=False,
            padding=1)
        self.conv9 = nn.Conv2d(64, 64, kernel_size=3, stride=2, bias=False,
            padding=1)
        self.conv10 = nn.Conv2d(64, 128, kernel_size=3, stride=2, bias=
            False, padding=1)
        self.conv11 = nn.Conv2d(128, 128, kernel_size=3, stride=2, bias=
            False, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.conv8(x)
        x = self.conv9(x)
        x = self.conv10(x)
        x = self.conv11(x)
        return x


def get_inputs():
    return [torch.rand([4, 3, 64, 64])]


def get_init_inputs():
    return [[], {}]
