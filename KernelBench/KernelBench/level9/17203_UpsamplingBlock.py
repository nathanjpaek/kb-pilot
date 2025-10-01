import torch
import torch.nn as nn


class UpsamplingBlock(nn.Module):

    def __init__(self, input_nc, output_nc, kernel, stride, pad):
        """
        Single block of upsampling operation

        Input:
        - int input_nc    : Input number of channels
        - int output_nc   : Output number of channels
        - int kernel      : Kernel size
        - int stride	  : Stride length
        - int pad         : Padd_moduleing
        """
        super(UpsamplingBlock, self).__init__()
        conv = nn.Conv2d
        biup = nn.Upsample
        block = nn.Sequential()
        block.add_module('conv_1', conv(input_nc, output_nc, kernel, stride,
            pad))
        block.add_module('upsample_2', biup(scale_factor=2, mode='bilinear'))
        self.biup_block = block

    def forward(self, x):
        return self.biup_block(x)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'input_nc': 4, 'output_nc': 4, 'kernel': 4, 'stride': 1,
        'pad': 4}]
