import torch
import torch.nn as nn
import torch.utils.data


class Conv2D(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, dilation_h
        =1, dilation_w=1, causal=True, use_wn_bias=True):
        super(Conv2D, self).__init__()
        self.causal = causal
        self.use_wn_bias = use_wn_bias
        self.dilation_h, self.dilation_w = dilation_h, dilation_w
        if self.causal:
            self.padding_h = dilation_h * (kernel_size - 1)
        else:
            self.padding_h = dilation_h * (kernel_size - 1) // 2
        self.padding_w = dilation_w * (kernel_size - 1) // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
            dilation=(dilation_h, dilation_w), padding=(self.padding_h,
            self.padding_w), bias=use_wn_bias)
        self.conv = nn.utils.weight_norm(self.conv)
        nn.init.kaiming_normal_(self.conv.weight)

    def forward(self, tensor):
        out = self.conv(tensor)
        if self.causal and self.padding_h != 0:
            out = out[:, :, :-self.padding_h, :]
        return out

    def reverse_fast(self, tensor):
        self.conv.padding = 0, self.padding_w
        out = self.conv(tensor)
        return out


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 4, 'out_channels': 4}]
