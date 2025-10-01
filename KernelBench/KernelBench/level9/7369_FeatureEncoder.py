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


class FeatureEncoder(nn.Module):

    def __init__(self, in_channel=34, out_channel=64):
        super(FeatureEncoder, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channel, out_channels=
            out_channel, kernel_size=3, stride=1, padding=1)
        self.act = nn.ReLU(inplace=True)
        self.resblock1 = ResBlock(out_channel, out_channel)
        self.resblock2 = ResBlock(out_channel, out_channel)

    def forward(self, batch):
        return self.resblock2(self.resblock1(self.act(self.conv(batch))))


def get_inputs():
    return [torch.rand([4, 34, 64, 64])]


def get_init_inputs():
    return [[], {}]
