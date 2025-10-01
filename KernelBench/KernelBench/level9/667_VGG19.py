import torch
import torch.nn as nn
import torch.nn.functional as F


class VGG19(nn.Module):

    def __init__(self):
        super(VGG19, self).__init__()
        self.conv0_0 = nn.Conv2d(in_channels=3, out_channels=64,
            kernel_size=3, stride=1, padding=1)
        self.conv0_1 = nn.Conv2d(in_channels=64, out_channels=64,
            kernel_size=3, stride=1, padding=1)
        self.conv1_0 = nn.Conv2d(in_channels=64, out_channels=128,
            kernel_size=3, stride=1, padding=1)
        self.conv1_1 = nn.Conv2d(in_channels=128, out_channels=128,
            kernel_size=3, stride=1, padding=1)
        self.conv2_0 = nn.Conv2d(in_channels=128, out_channels=256,
            kernel_size=3, stride=1, padding=1)
        self.conv2_1 = nn.Conv2d(in_channels=256, out_channels=256,
            kernel_size=3, stride=1, padding=1)
        self.conv2_2 = nn.Conv2d(in_channels=256, out_channels=256,
            kernel_size=3, stride=1, padding=1)
        self.conv3_0 = nn.Conv2d(in_channels=256, out_channels=512,
            kernel_size=3, stride=1, padding=1)
        self.conv3_1 = nn.Conv2d(in_channels=512, out_channels=512,
            kernel_size=3, stride=1, padding=1)
        self.conv3_2 = nn.Conv2d(in_channels=512, out_channels=512,
            kernel_size=3, stride=1, padding=1)
        self.conv4_0 = nn.Conv2d(in_channels=512, out_channels=512,
            kernel_size=3, stride=1, padding=1)
        self.conv4_1 = nn.Conv2d(in_channels=512, out_channels=512,
            kernel_size=3, stride=1, padding=1)
        self.conv4_2 = nn.Conv2d(in_channels=512, out_channels=512,
            kernel_size=3, stride=1, padding=1)
        self.fc = nn.Linear(512, 10)

    def forward(self, x):
        x = F.relu(self.conv0_0(x))
        x = F.relu(self.conv0_1(x))
        x = nn.MaxPool2d(2, 2)(x)
        x = F.relu(self.conv1_0(x))
        x = F.relu(self.conv1_1(x))
        x = nn.MaxPool2d(2, 2)(x)
        x = F.relu(self.conv2_0(x))
        x = F.relu(self.conv2_1(x))
        x = F.relu(self.conv2_2(x))
        x = nn.MaxPool2d(2, 2)(x)
        x = F.relu(self.conv3_0(x))
        x = F.relu(self.conv3_1(x))
        x = F.relu(self.conv3_2(x))
        x = nn.MaxPool2d(2, 2)(x)
        x = F.relu(self.conv4_0(x))
        x = F.relu(self.conv4_1(x))
        x = F.relu(self.conv4_2(x))
        _b, _c, h, _w = x.size()
        x = nn.AvgPool2d(h)(x)
        x = nn.Flatten()(x)
        x = self.fc(x)
        return x


def get_inputs():
    return [torch.rand([4, 3, 64, 64])]


def get_init_inputs():
    return [[], {}]
