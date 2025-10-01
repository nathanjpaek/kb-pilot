import torch
from torch import nn
import torch.utils.data
import torch.nn.functional as F


class GatedMaskedConv2d(nn.Module):

    def __init__(self, in_dim, out_dim=None, kernel_size=3, mask='B'):
        super(GatedMaskedConv2d, self).__init__()
        if out_dim is None:
            out_dim = in_dim
        self.dim = out_dim
        self.size = kernel_size
        self.mask = mask
        pad = self.size // 2
        self.v_conv = nn.Conv2d(in_dim, 2 * self.dim, kernel_size=(pad + 1,
            self.size))
        self.v_pad1 = nn.ConstantPad2d((pad, pad, pad, 0), 0)
        self.v_pad2 = nn.ConstantPad2d((0, 0, 1, 0), 0)
        self.vh_conv = nn.Conv2d(2 * self.dim, 2 * self.dim, kernel_size=1)
        self.h_conv = nn.Conv2d(in_dim, 2 * self.dim, kernel_size=(1, pad + 1))
        self.h_pad1 = nn.ConstantPad2d((self.size // 2, 0, 0, 0), 0)
        self.h_pad2 = nn.ConstantPad2d((1, 0, 0, 0), 0)
        self.h_conv_res = nn.Conv2d(self.dim, self.dim, 1)

    def forward(self, v_map, h_map):
        v_out = self.v_pad2(self.v_conv(self.v_pad1(v_map)))[:, :, :-1, :]
        v_map_out = F.tanh(v_out[:, :self.dim]) * F.sigmoid(v_out[:, self.dim:]
            )
        vh = self.vh_conv(v_out)
        h_out = self.h_conv(self.h_pad1(h_map))
        if self.mask == 'A':
            h_out = self.h_pad2(h_out)[:, :, :, :-1]
        h_out = h_out + vh
        h_out = F.tanh(h_out[:, :self.dim]) * F.sigmoid(h_out[:, self.dim:])
        h_map_out = self.h_conv_res(h_out)
        if self.mask == 'B':
            h_map_out = h_map_out + h_map
        return v_map_out, h_map_out


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_dim': 4}]
