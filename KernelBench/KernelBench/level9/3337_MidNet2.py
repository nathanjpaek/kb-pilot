import torch
import torch.nn as nn


class MidNet2(nn.Module):

    def forward(self, x_in):
        """Network with dilation rate 2

        :param x_in: input convolutional features        
        :returns: processed convolutional features        
        :rtype: Tensor

        """
        x = self.lrelu(self.conv1(x_in))
        x = self.lrelu(self.conv2(x))
        x = self.lrelu(self.conv3(x))
        x = self.conv4(x)
        return x

    def __init__(self, in_channels=16):
        """FIXME! briefly describe function

        :param in_channels: Input channels
        :returns: N/A
        :rtype: N/A

        """
        super(MidNet2, self).__init__()
        self.lrelu = nn.LeakyReLU()
        self.conv1 = nn.Conv2d(in_channels, 64, 3, 1, 2, 2)
        self.conv2 = nn.Conv2d(64, 64, 3, 1, 2, 2)
        self.conv3 = nn.Conv2d(64, 64, 3, 1, 2, 2)
        self.conv4 = nn.Conv2d(64, 64, 3, 1, 2, 2)


def get_inputs():
    return [torch.rand([4, 16, 64, 64])]


def get_init_inputs():
    return [[], {}]
