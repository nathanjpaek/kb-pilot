import torch
import torch.nn as nn
import torch.nn.functional as F


class WeightNet_DW(nn.Module):
    """ Here we show a grouping manner when we apply
    WeightNet to a depthwise convolution. The grouped
    fc layer directly generates the convolutional kernel,
    has fewer parameters while achieving comparable results.
    This layer has M/G*inp inputs, inp groups and inp*ksize*ksize outputs.

    Args:
        inp (int): Number of input channels
        oup (int): Number of output channels
        ksize (int): Size of the convolving kernel
        stride (int): Stride of the convolution
    """

    def __init__(self, inp, ksize, stride):
        super().__init__()
        self.M = 2
        self.G = 2
        self.pad = ksize // 2
        inp_gap = max(16, inp // 16)
        self.inp = inp
        self.ksize = ksize
        self.stride = stride
        self.wn_fc1 = nn.Conv2d(inp_gap, self.M // self.G * inp, 1, 1, 0,
            groups=1, bias=True)
        self.sigmoid = nn.Sigmoid()
        self.wn_fc2 = nn.Conv2d(self.M // self.G * inp, inp * ksize * ksize,
            1, 1, 0, groups=inp, bias=False)

    def forward(self, x, x_gap):
        """ Input:
            x (bs*c*h*w): the output feature from previous convolution layer
            x_gap (bs*inp_gap*1*1): the output feature from reduction layer
        """
        x_w = self.wn_fc1(x_gap)
        x_w = self.sigmoid(x_w)
        x_w = self.wn_fc2(x_w)
        batch_size = x.shape[0]
        x = x.reshape(1, -1, x.shape[2], x.shape[3])
        x_w = x_w.reshape(-1, 1, self.ksize, self.ksize)
        x = F.conv2d(x, weight=x_w, stride=self.stride, padding=self.pad,
            groups=batch_size * self.inp)
        x = x.reshape(-1, self.inp, x.shape[2], x.shape[3])
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 16, 64, 64])]


def get_init_inputs():
    return [[], {'inp': 4, 'ksize': 4, 'stride': 1}]
