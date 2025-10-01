import torch
import torch.nn as nn
import torch.nn.functional


class OutPutBlock(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(OutPutBlock, self).__init__()
        self.in_chns = in_channels
        self.out_chns = out_channels
        self.conv1 = nn.Conv2d(self.in_chns, self.in_chns // 2, kernel_size
            =1, padding=0)
        self.conv2 = nn.Conv2d(self.in_chns // 2, self.out_chns,
            kernel_size=1, padding=0)
        self.drop1 = nn.Dropout2d(0.3)
        self.drop2 = nn.Dropout2d(0.3)
        self.ac1 = nn.LeakyReLU()

    def forward(self, x):
        x = self.drop1(x)
        x = self.conv1(x)
        x = self.ac1(x)
        x = self.drop2(x)
        x = self.conv2(x)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 4, 'out_channels': 4}]
