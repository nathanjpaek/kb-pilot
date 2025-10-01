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


class stage_1_block(nn.Module):
    """
    stage 1 only 5 layers and the kernal size is 5
    last layer don't have relu
    """

    def __init__(self, input_dim, output_dim):
        super(stage_1_block, self).__init__()
        self.conv1 = conv(input_dim, 128, 3, 1, 1)
        self.conv2 = conv(128, 128, 3, 1, 1)
        self.conv3 = conv(128, 128, 3, 1, 1)
        self.conv4 = conv(128, 512, 1, 1, 0)
        self.conv5 = nn.Conv2d(512, output_dim, 1, 1, 0)
        self.initi()

    def forward(self, input_):
        output = self.conv1(input_)
        output = self.conv2(output)
        output = self.conv3(output)
        output = self.conv4(output)
        output = self.conv5(output)
        return output

    def initi(self):
        init.normal_(self.conv5.weight, std=0.01)
        if self.conv5.bias is not None:
            init.constant_(self.conv5.bias, 0.0)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'input_dim': 4, 'output_dim': 4}]
