import torch
import torch.nn as nn


class ConvElu(nn.Module):

    def __init__(self, in_ch=3, out_ch=3, dirate=1):
        super(ConvElu, self).__init__()
        self.conv_s1 = nn.Conv2d(in_ch, out_ch, 3, padding=1 * dirate,
            dilation=1 * dirate, padding_mode='reflect')
        self.elu = nn.ELU(inplace=True)

    def forward(self, x):
        hx = x
        xout = self.elu(self.conv_s1(hx))
        return xout


def get_inputs():
    return [torch.rand([4, 3, 4, 4])]


def get_init_inputs():
    return [[], {}]
