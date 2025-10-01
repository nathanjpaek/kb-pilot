import torch
from torch import nn


class CoordConv2D(nn.Module):

    def __init__(self, in_channels: 'int', out_channels: 'int', kernel_size:
        'int'=3, stride: 'int'=1, padding: 'int'=1, with_r: 'bool'=False):
        super().__init__()
        self.in_channel = in_channels
        self.with_r = with_r
        self.conv = nn.Conv2d(in_channels=in_channels + (2 if not with_r else
            3), out_channels=out_channels, kernel_size=kernel_size, stride=
            stride, padding=padding)

    def forward(self, input_tensor: 'torch.Tensor'):
        batch_size, _, y_dim, x_dim = input_tensor.size()
        xx_channel = torch.arange(x_dim).repeat(1, y_dim, 1)
        yy_channel = torch.arange(y_dim).repeat(1, x_dim, 1).transpose(1, 2)
        xx_channel = xx_channel.float() / (x_dim - 1) * 2.0 - 1.0
        yy_channel = yy_channel.float() / (y_dim - 1) * 2.0 - 1.0
        xx_channel = xx_channel.repeat(batch_size, 1, 1, 1)
        yy_channel = yy_channel.repeat(batch_size, 1, 1, 1)
        x = torch.cat([input_tensor, xx_channel.type_as(input_tensor),
            yy_channel.type_as(input_tensor)], dim=1)
        if self.with_r:
            rr = torch.sqrt(torch.pow(xx_channel.type_as(input_tensor) - 
                0.5, 2) + torch.pow(yy_channel.type_as(input_tensor) - 0.5, 2))
            x = torch.cat([x, rr], dim=1)
        x = self.conv(x)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 4, 'out_channels': 4}]
