import torch
import torch.nn as nn


class srcEncoder(nn.Module):

    def __init__(self, in_ch, hid_ch):
        super(srcEncoder, self).__init__()
        self.act = nn.ReLU()
        self.conv1 = nn.Conv2d(in_ch, hid_ch, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(hid_ch, hid_ch, kernel_size=3, padding=1)

    def forward(self, x):
        return self.conv2(self.act(self.conv1(x)))


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_ch': 4, 'hid_ch': 4}]
