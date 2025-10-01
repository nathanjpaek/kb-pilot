import torch
import numpy as np
import torch.serialization
import torch
import torch.nn as nn
import torch.utils.data


class ConvLayer(nn.Module):

    def __init__(self, in_dim, out_dim, kernel_size, stride, dilation=1):
        super(ConvLayer, self).__init__()
        self.dilation = dilation
        if dilation == 1:
            reflect_padding = int(np.floor(kernel_size / 2))
            self.reflection_pad = nn.ReflectionPad2d(reflect_padding)
            self.conv2d = nn.Conv2d(in_dim, out_dim, kernel_size, stride,
                dilation=dilation)
        else:
            self.conv2d = nn.Conv2d(in_dim, out_dim, kernel_size, stride,
                dilation=dilation, padding=dilation)

    def forward(self, x):
        if self.dilation == 1:
            out = self.reflection_pad(x)
            out = self.conv2d(out)
        else:
            out = self.conv2d(x)
        return out


class DuRB_p(nn.Module):

    def __init__(self, in_dim=32, out_dim=32, res_dim=32, k1_size=3,
        k2_size=1, dilation=1, norm_type='batch_norm', with_relu=True):
        super(DuRB_p, self).__init__()
        self.conv1 = ConvLayer(in_dim, in_dim, 3, 1)
        self.conv2 = ConvLayer(in_dim, in_dim, 3, 1)
        self.up_conv = ConvLayer(in_dim, res_dim, kernel_size=k1_size,
            stride=1, dilation=dilation)
        self.down_conv = ConvLayer(res_dim, out_dim, kernel_size=k2_size,
            stride=1)
        self.with_relu = with_relu
        self.relu = nn.ReLU()

    def forward(self, x, res):
        x_r = x
        x = self.relu(self.conv1(x))
        x = self.conv2(x)
        x += x_r
        x = self.relu(x)
        x = self.up_conv(x)
        x += res
        x = self.relu(x)
        res = x
        x = self.down_conv(x)
        x += x_r
        if self.with_relu:
            x = self.relu(x)
        else:
            pass
        return x, res


def get_inputs():
    return [torch.rand([4, 32, 4, 4]), torch.rand([4, 32, 4, 4])]


def get_init_inputs():
    return [[], {}]
