import math
import torch
import torch.utils.data
import torch
import torch.nn as nn


class Conv2dSame(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
        dilation=1):
        super(Conv2dSame, self).__init__()
        self.F = kernel_size
        self.S = stride
        self.D = dilation
        self.layer = nn.Conv2d(in_channels, out_channels, kernel_size,
            stride, dilation=dilation)

    def forward(self, x_in):
        _N, _C, H, W = x_in.shape
        H2 = math.ceil(H / self.S)
        W2 = math.ceil(W / self.S)
        Pr = (H2 - 1) * self.S + (self.F - 1) * self.D + 1 - H
        Pc = (W2 - 1) * self.S + (self.F - 1) * self.D + 1 - W
        x_pad = nn.ZeroPad2d((Pr // 2, Pr - Pr // 2, Pc // 2, Pc - Pc // 2))(
            x_in)
        x_relu = nn.ReLU()(x_pad)
        x_out = self.layer(x_relu)
        return x_out


class FeatureExtractionBlock(nn.Module):
    """Initialize the Resnet block

    A resnet block is a conv block with skip connections
    We construct a conv block with build_conv_block function,
    and implement skip connections in <forward> function.
    Original Resnet paper: https://arxiv.org/pdf/1512.03385.pdf
    """

    def __init__(self, input_nc, ngf=64, norm_layer=nn.BatchNorm2d,
        use_dropout=False):
        """Construct a convolutional block.

        Parameters:
            dim (int)           -- the number of channels in the conv layer.
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
            use_bias (bool)     -- if the conv layer uses bias or not

        Returns a conv block (with a conv layer, a normalization layer, and a non-linearity layer (ReLU))
        """
        super(FeatureExtractionBlock, self).__init__()
        self.C3R1 = Conv2dSame(input_nc, ngf, kernel_size=3)
        self.C1R1 = Conv2dSame(ngf, ngf, kernel_size=1)
        self.C3R2 = Conv2dSame(input_nc, ngf, kernel_size=3)
        self.C1R2 = Conv2dSame(ngf, ngf, kernel_size=1)

    def forward(self, input):
        return self.C3R1(input) + self.C3R2(input) + self.C1R1(self.C3R1(input)
            ) + self.C1R2(self.C3R2(input))


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'input_nc': 4}]
