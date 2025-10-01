import torch
import torch.nn as nn
from torch.nn import init


class conv(nn.Module):
    """
    n*n conv with relu
    """

    def __init__(self, in_dim, out_dim, kernal_size, stride, padding):
        super(conv, self).__init__()
        self.con_layer = nn.Conv2d(in_dim, out_dim, kernal_size, stride,
            padding)
        self.relu = nn.ReLU(inplace=True)
        self.initi()

    def forward(self, input_):
        output = self.con_layer(input_)
        output = self.relu(output)
        return output

    def initi(self):
        init.normal_(self.con_layer.weight, std=0.01)
        if self.con_layer.bias is not None:
            init.constant_(self.con_layer.bias, 0.0)


class VGG_19(nn.Module):
    """
    VGG_19 first 10 layers
    11 and 12 by CMU
    """

    def __init__(self, input_dim):
        super(VGG_19, self).__init__()
        self.conv1_1 = conv(input_dim, 64, 3, 1, 1)
        self.conv1_2 = conv(64, 64, 3, 1, 1)
        self.pooling_1 = nn.MaxPool2d(2, 2, 0)
        self.conv2_1 = conv(64, 128, 3, 1, 1)
        self.conv2_2 = conv(128, 128, 3, 1, 1)
        self.pooling_2 = nn.MaxPool2d(2, 2, 0)
        self.conv3_1 = conv(128, 256, 3, 1, 1)
        self.conv3_2 = conv(256, 256, 3, 1, 1)
        self.conv3_3 = conv(256, 256, 3, 1, 1)
        self.conv3_4 = conv(256, 256, 3, 1, 1)
        self.pooling_3 = nn.MaxPool2d(2, 2, 0)
        self.conv4_1 = conv(256, 512, 3, 1, 1)
        self.conv4_2 = conv(512, 512, 3, 1, 1)
        self.conv4_3 = conv(512, 256, 3, 1, 1)
        self.conv4_4 = conv(256, 128, 3, 1, 1)

    def forward(self, input_):
        output = self.conv1_1(input_)
        output = self.conv1_2(output)
        output = self.pooling_1(output)
        output = self.conv2_1(output)
        output = self.conv2_2(output)
        output = self.pooling_2(output)
        output = self.conv3_1(output)
        output = self.conv3_2(output)
        output = self.conv3_3(output)
        output = self.conv3_4(output)
        output = self.pooling_3(output)
        output = self.conv4_1(output)
        output = self.conv4_2(output)
        output = self.conv4_3(output)
        output = self.conv4_4(output)
        return output


def get_inputs():
    return [torch.rand([4, 4, 64, 64])]


def get_init_inputs():
    return [[], {'input_dim': 4}]
