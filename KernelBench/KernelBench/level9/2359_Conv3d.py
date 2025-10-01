import torch
import torch.nn as nn


class Conv3d(nn.Module):
    """
    This class is for a convolutional layer.
    3d卷积
    """

    def __init__(self, nIn, nOut, kSize, stride=1):
        """
        :param nIn: number of input channels
        :param nOut: number of output channels
        :param kSize: kernel size
        :param stride: optional stride rate for down-sampling
        pytorch
        in N, Ci, D, H, W
        out N, Co, D, H, W
        tensorflow
        [batch, in_depth, in_height, in_width, in_channels] N,D,H,W,C
        """
        super().__init__()
        padding = int((kSize - 1) / 2)
        self.conv = nn.Conv3d(nIn, nOut, (kSize, kSize, kSize), stride=
            stride, padding=(padding, padding, padding), bias=True)

    def forward(self, input):
        """
        :param input: input feature map
        :return: transformed feature map
        """
        output = self.conv(input)
        return output


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'nIn': 4, 'nOut': 4, 'kSize': 4}]
