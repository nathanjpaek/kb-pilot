import torch
import torch.nn as nn


def projection_pooling_column(input):
    b, c, _h, w = input.size()
    input = input.permute(0, 1, 3, 2)
    ave_v = input.mean(dim=3)
    ave_v = ave_v.reshape(b, c, w, -1)
    input[:, :, :, :] = ave_v[:, :, :]
    input = input.permute(0, 1, 3, 2)
    return input


def projection_pooling_row(input):
    b, c, h, _w = input.size()
    ave_v = input.mean(dim=3)
    ave_v = ave_v.reshape(b, c, h, -1)
    input[:, :, :, :] = ave_v[:, :, :]
    return input


class Block(nn.Module):

    def __init__(self, in_channels, i, row_column=0):
        super(Block, self).__init__()
        self.index = i
        self.row_column = row_column
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=6,
            kernel_size=3, padding=2, dilation=2)
        self.conv2 = nn.Conv2d(in_channels=in_channels, out_channels=6,
            kernel_size=3, padding=3, dilation=3)
        self.conv3 = nn.Conv2d(in_channels=in_channels, out_channels=6,
            kernel_size=3, padding=4, dilation=4)
        self.pool1 = nn.MaxPool2d(kernel_size=(1, 2), stride=1)
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 1), stride=1)
        self.branch1 = nn.Conv2d(in_channels=18, out_channels=18, kernel_size=1
            )
        self.branch2 = nn.Conv2d(in_channels=18, out_channels=1, kernel_size=1)

    def forward(self, input):
        out1 = torch.cat([self.conv1(input), self.conv2(input), self.conv3(
            input)], dim=1)
        if self.index <= 3:
            if self.row_column == 0:
                out1 = self.pool1(out1)
            else:
                out1 = self.pool2(out1)
        if self.row_column == 0:
            b1 = projection_pooling_row(self.branch1(out1))
            b2 = projection_pooling_row(self.branch2(out1))
        else:
            b1 = projection_pooling_column(self.branch1(out1))
            b2 = projection_pooling_column(self.branch2(out1))
        _b, _c, _h, _w = b2.size()
        b2 = torch.sigmoid(b2)
        output = torch.cat([b1, out1, b2], dim=1)
        return output, b2


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 4, 'i': 4}]
