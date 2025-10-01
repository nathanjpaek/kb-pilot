import torch
from torch import nn
import torch.nn.functional as F


class discriminator2(nn.Module):

    def __init__(self):
        super().__init__()
        self.d1 = nn.Conv2d(in_channels=1, out_channels=4, kernel_size=3,
            stride=1, padding=1)
        self.d2 = nn.Conv2d(in_channels=4, out_channels=8, kernel_size=3,
            stride=1, padding=1)
        self.d3 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3,
            stride=1, padding=1)
        self.d4 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3,
            stride=1, padding=1)
        self.val = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=1,
            stride=1, padding=0)
        self.maxpool = nn.MaxPool2d(2)

    def forward(self, x):
        x = F.leaky_relu(self.d1(x), 0.2)
        x = self.maxpool(x)
        x = F.instance_norm(F.leaky_relu(self.d2(x), 0.2))
        x = self.maxpool(x)
        x = F.instance_norm(F.leaky_relu(self.d3(x), 0.2))
        x = self.maxpool(x)
        x = F.instance_norm(F.leaky_relu(self.d4(x), 0.2))
        x = self.maxpool(x)
        x = self.val(x)
        x = F.sigmoid(x)
        return x


def get_inputs():
    return [torch.rand([4, 1, 64, 64])]


def get_init_inputs():
    return [[], {}]
