import torch
import torch.nn as nn


class CustomConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
        padding=None, bias=True, residual_init=True):
        super(CustomConv2d, self).__init__()
        self.residual_init = residual_init
        if padding is None:
            padding = int((kernel_size - 1) / 2)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, bias=bias)

    def forward(self, input):
        return self.conv(input)


class DepthToSpace(nn.Module):

    def __init__(self, block_size):
        super(DepthToSpace, self).__init__()
        self.block_size = block_size
        self.block_size_square = block_size * block_size

    def forward(self, input):
        output = input.permute(0, 2, 3, 1)
        batch_size, in_height, in_width, in_depth = output.size()
        out_depth = int(in_depth / self.block_size_square)
        out_width = int(in_width * self.block_size)
        out_height = int(in_height * self.block_size)
        output = output.contiguous().view(batch_size, in_height, in_width,
            self.block_size_square, out_depth)
        output_list = output.split(self.block_size, 3)
        output_list = [output_element.contiguous().view(batch_size,
            in_height, out_width, out_depth) for output_element in output_list]
        output = torch.stack(output_list, 0).transpose(0, 1).permute(0, 2, 
            1, 3, 4).contiguous().view(batch_size, out_height, out_width,
            out_depth)
        output = output.permute(0, 3, 1, 2)
        return output


class UpSampleConv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, bias=True,
        residual_init=True):
        super(UpSampleConv, self).__init__()
        self.conv = CustomConv2d(in_channels, out_channels, kernel_size,
            bias=bias, residual_init=residual_init)
        self.depth_to_space = DepthToSpace(2)

    def forward(self, input):
        output = input
        output = torch.cat((output, output, output, output), 1)
        output = self.depth_to_space(output)
        output = self.conv(output)
        return output


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}]
