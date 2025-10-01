import torch
import torch.nn as nn


def get_same_padding(kernel_size, dilation):
    kernel_size = kernel_size + (kernel_size - 1) * (dilation - 1)
    padding = (kernel_size - 1) // 2
    return padding


class BinResBlock(nn.Module):

    def __init__(self, inplanes, kernel_size=3, dilation=1):
        super(BinResBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes * 2, inplanes, kernel_size=
            kernel_size, stride=1, padding=get_same_padding(kernel_size,
            dilation), dilation=dilation)
        self.conv2 = nn.Conv2d(inplanes * 2, inplanes, kernel_size=
            kernel_size, stride=1, padding=get_same_padding(kernel_size,
            dilation), dilation=dilation)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, y, H_pre):
        residual = H_pre
        x_out = self.relu(self.conv1(torch.cat([x, H_pre], dim=1)))
        y_out = self.relu(self.conv1(torch.cat([y, H_pre], dim=1)))
        H_out = self.conv2(torch.cat([x_out, y_out], dim=1))
        H_out += residual
        return x_out, y_out, H_out


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand(
        [4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'inplanes': 4}]
