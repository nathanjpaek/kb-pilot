import torch
from torch import Tensor
from typing import List
from typing import Tuple
from typing import Union
import torch.nn as nn


class DownSamplingBlock(nn.Module):

    def __init__(self, in_channels: 'int', channel_up_factor: 'int'=2,
        max_pooling: 'bool'=True, dropout: 'Tuple'=(0, 0)):
        super().__init__()
        out_channels = in_channels * channel_up_factor
        self.max_pooling = max_pooling
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
            stride=1, padding=1)
        self.activation1 = nn.ReLU()
        self.dropout1 = nn.Dropout2d(dropout[0])
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
            stride=1, padding=1)
        self.activation2 = nn.ReLU()
        self.dropout2 = nn.Dropout2d(dropout[1])
        if self.max_pooling:
            self.max = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x: 'Tensor') ->Union[Tensor, Tuple[Tensor]]:
        x = self.conv1(x)
        x = self.activation1(x)
        x = self.dropout1(x)
        x = self.conv2(x)
        x = self.activation2(x)
        x = self.dropout2(x)
        skip_connection = x
        if self.max_pooling:
            x = self.max(x)
            return x, skip_connection
        return x


class UpSamplingBlock(nn.Module):

    def __init__(self, in_channels: 'int', channel_down_factor: 'int',
        skip_channels: 'int', dropout: 'Tuple'=(0, 0, 0)):
        super().__init__()
        out_channels = in_channels // channel_down_factor
        self.transpose_conv2d = nn.ConvTranspose2d(in_channels,
            out_channels, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.dropout = nn.Dropout2d(dropout[0])
        self.conv1 = nn.Conv2d(out_channels + skip_channels, out_channels,
            kernel_size=3, stride=1, padding=1)
        self.activation1 = nn.ReLU()
        self.dropout1 = nn.Dropout2d(dropout[1])
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
            stride=1, padding=1)
        self.activation2 = nn.ReLU()
        self.dropout2 = nn.Dropout2d(dropout[2])

    def forward(self, x: 'Tensor', skip_connection: 'Tensor') ->Tensor:
        x = self.transpose_conv2d(x)
        x = torch.cat([x, skip_connection], -3)
        x = self.dropout(x)
        x = self.conv1(x)
        x = self.activation1(x)
        x = self.dropout1(x)
        x = self.conv2(x)
        x = self.activation2(x)
        x = self.dropout2(x)
        return x


class BuildingsModel(nn.Module):

    def __init__(self, in_channels: 'int', init_out_factor: 'int', dropouts:
        'List'=[0] * 23):
        super().__init__()
        self.hooks = {}
        self.down_1 = DownSamplingBlock(in_channels, init_out_factor,
            dropout=(dropouts[0], dropouts[1]))
        self.down_2 = DownSamplingBlock(self.down_1.conv2.out_channels, 2,
            dropout=(dropouts[2], dropouts[3]))
        self.down_3 = DownSamplingBlock(self.down_2.conv2.out_channels, 2,
            dropout=(dropouts[4], dropouts[5]))
        self.down_4 = DownSamplingBlock(self.down_3.conv2.out_channels, 2,
            dropout=(dropouts[6], dropouts[7]))
        self.down_5 = DownSamplingBlock(self.down_4.conv2.out_channels, 2,
            max_pooling=False, dropout=(dropouts[8], dropouts[9]))
        self.up_1 = UpSamplingBlock(self.down_5.conv2.out_channels, 2,
            skip_channels=self.down_4.conv2.out_channels, dropout=(dropouts
            [10], dropouts[11], dropouts[12]))
        self.up_2 = UpSamplingBlock(self.up_1.conv2.out_channels, 2,
            skip_channels=self.down_3.conv2.out_channels, dropout=(dropouts
            [13], dropouts[14], dropouts[15]))
        self.up_3 = UpSamplingBlock(self.up_2.conv2.out_channels, 2,
            skip_channels=self.down_2.conv2.out_channels, dropout=(dropouts
            [16], dropouts[17], dropouts[18]))
        self.up_4 = UpSamplingBlock(self.up_3.conv2.out_channels, 2,
            skip_channels=self.down_1.conv2.out_channels, dropout=(dropouts
            [19], dropouts[20], dropouts[21]))
        self.zconv = nn.Conv2d(self.up_4.conv2.out_channels, out_channels=2,
            kernel_size=1)
        self.prob = nn.Softmax(dim=-3)
        self.__init_weights__()

    def __init_weights__(self):
        for m in self.modules():
            if type(m) in {nn.Conv2d, nn.ConvTranspose2d}:
                nn.init.kaiming_normal_(m.weight.data, mode='fan_in',
                    nonlinearity='relu')
                if m.bias is not None:
                    m.bias.data.fill_(0)

    def forward(self, x):
        x, skip_connection_4 = self.down_1(x)
        x, skip_connection_3 = self.down_2(x)
        x, skip_connection_2 = self.down_3(x)
        x, skip_connection_1 = self.down_4(x)
        x = self.down_5(x)
        x = self.up_1(x, skip_connection_1)
        x = self.up_2(x, skip_connection_2)
        x = self.up_3(x, skip_connection_3)
        x = self.up_4(x, skip_connection_4)
        z = self.zconv(x)
        a = self.prob(z)
        return z, a


def get_inputs():
    return [torch.rand([4, 4, 64, 64])]


def get_init_inputs():
    return [[], {'in_channels': 4, 'init_out_factor': 4}]
