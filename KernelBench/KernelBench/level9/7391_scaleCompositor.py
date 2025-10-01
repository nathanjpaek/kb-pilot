import torch
import torch.nn as nn


class ResBlock(nn.Module):

    def __init__(self, in_ch, hid_ch):
        super(ResBlock, self).__init__()
        self.act = nn.ReLU()
        self.conv1 = nn.Conv2d(in_ch, hid_ch, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(hid_ch, hid_ch, kernel_size=3, padding=1)

    def forward(self, x):
        return x + self.act(self.conv2(self.act(self.conv1(x))))


class scaleCompositor(nn.Module):

    def __init__(self, in_ch, hid_ch):
        super(scaleCompositor, self).__init__()
        self.conv1 = nn.Conv2d(in_ch, hid_ch, kernel_size=1)
        self.conv2 = nn.Conv2d(hid_ch, 1, kernel_size=1)
        self.resblock1 = ResBlock(hid_ch, hid_ch)
        self.resblock2 = ResBlock(hid_ch, hid_ch)
        self.act = nn.Sigmoid()
        self.downsample = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, f, c):
        x = torch.cat((f, c), dim=1)
        UDf = self.upsample(self.downsample(f))
        scale = self.conv2(self.resblock2(self.resblock1(self.conv1(x))))
        scale = self.act(scale)
        return f - torch.matmul(scale, UDf) + torch.matmul(scale, c)


def get_inputs():
    return [torch.rand([4, 1, 4, 4]), torch.rand([4, 3, 4, 4])]


def get_init_inputs():
    return [[], {'in_ch': 4, 'hid_ch': 4}]
