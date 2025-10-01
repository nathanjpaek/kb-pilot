import torch
import torch.nn as nn
from itertools import product as product


class MaskNet(nn.Module):

    def __init__(self):
        super(MaskNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=
            5, stride=1, padding=2)
        self.relu1 = nn.ReLU()
        self.Pool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size
            =3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.conv2s = nn.Conv2d(in_channels=32, out_channels=2, kernel_size
            =1, stride=1, padding=0)
        self.Pool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size
            =(3, 3), stride=1, padding=1)
        self.relu3 = nn.ReLU()
        self.conv3s = nn.Conv2d(in_channels=64, out_channels=2, kernel_size
            =1, stride=1, padding=0)
        self.Pool3 = nn.MaxPool2d(kernel_size=(4, 4), stride=4)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=128,
            kernel_size=(3, 3), stride=1, padding=1)
        self.relu4 = nn.ReLU()
        self.conv4s = nn.Conv2d(in_channels=128, out_channels=2,
            kernel_size=1, stride=1, padding=0)
        self.Upsample1 = nn.ConvTranspose2d(in_channels=2, out_channels=2,
            kernel_size=8, stride=4, padding=2, bias=False, groups=2)
        self.Upsample2 = nn.ConvTranspose2d(in_channels=2, out_channels=2,
            kernel_size=4, stride=2, padding=1, bias=False, groups=2)
        self.Upsample3 = nn.ConvTranspose2d(in_channels=2, out_channels=2,
            kernel_size=4, stride=2, padding=1, bias=False, groups=2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.Pool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        branch1 = self.conv2s(x)
        x = self.Pool2(x)
        x = self.conv3(x)
        x = self.relu3(x)
        branch2 = self.conv3s(x)
        x = self.Pool3(x)
        x = self.conv4(x)
        x = self.relu4(x)
        x = self.conv4s(x)
        branch3 = self.Upsample1(x)
        branch_intermediate1 = branch2 + branch3
        branch_intermediate2 = self.Upsample2(branch_intermediate1)
        branch_intermediate3 = branch_intermediate2 + branch1
        output = self.Upsample3(branch_intermediate3)
        return output


def get_inputs():
    return [torch.rand([4, 1, 64, 64])]


def get_init_inputs():
    return [[], {}]
