import time
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from time import time as time


class Cnv2d_separable(nn.Module):

    def __init__(self, n_input_ch, n_output_ch, kernel_size, stride,
        padding, bias=False, red_portion=0.5):
        super(Cnv2d_separable, self).__init__()
        self.n_input_ch = n_input_ch
        self.n_input_ch_red = int(n_input_ch * red_portion)
        self.n_output_ch = n_output_ch
        self.n_output_ch_red = int(n_output_ch * red_portion)
        self.n_output_ch_green = n_output_ch - self.n_output_ch_red
        self.conv_half = nn.Conv2d(self.n_input_ch_red, self.
            n_output_ch_red, kernel_size, stride, padding, bias=bias)
        self.conv_all = nn.Conv2d(self.n_input_ch, self.n_output_ch_green,
            kernel_size, stride, padding, bias=bias)

    def forward(self, input):
        first_half = input[:, :self.n_input_ch_red, :, :]
        first_half_conv = self.conv_half(first_half)
        full_conv = self.conv_all(input)
        all_conv = torch.cat((first_half_conv, full_conv), 1)
        return all_conv


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'n_input_ch': 4, 'n_output_ch': 4, 'kernel_size': 4,
        'stride': 1, 'padding': 4}]
