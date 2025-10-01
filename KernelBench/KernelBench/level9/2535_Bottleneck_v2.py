import torch
import torch.nn as nn
import torch.cuda
import torch.backends.cudnn
import torch.backends.mkl


class Bottleneck_v2(nn.Module):

    def __init__(self):
        super(Bottleneck_v2, self).__init__()
        self.conv = nn.Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1),
            bias=True)
        self.conv1 = nn.Conv2d(256, 64, kernel_size=(1, 1), stride=(1, 1),
            bias=True)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1),
            padding=(1, 1), bias=True)
        self.conv3 = nn.Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1),
            bias=True)

    def forward(self, x):
        x = self.conv(x)
        y1 = self.conv1(x).relu_()
        y2 = self.conv2(y1).relu_()
        y3 = self.conv3(y2)
        y3 += x
        return y3.relu_()


def get_inputs():
    return [torch.rand([4, 64, 64, 64])]


def get_init_inputs():
    return [[], {}]
