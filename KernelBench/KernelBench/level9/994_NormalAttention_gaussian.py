import torch
import torch.nn as nn


class NormalAttention_gaussian(nn.Module):

    def __init__(self, input_channel_num):
        super(NormalAttention_gaussian, self).__init__()
        self.c_in = input_channel_num
        self.value_conv = nn.Conv2d(in_channels=self.c_in, out_channels=
            self.c_in, kernel_size=1)
        self.gamma = nn.Conv2d(in_channels=self.c_in, out_channels=self.
            c_in, kernel_size=1)

    def forward(self, x):
        B, C, H, W = x.size()
        proj_query = x.view(B, -1, H * W).permute(0, 2, 1)
        proj_key = x.view(B, -1, H * W)
        energy = torch.bmm(proj_query, proj_key)
        energy = torch.exp(energy)
        energy_sum = torch.sum(energy, dim=2, keepdim=True)
        energy = energy / energy_sum
        proj_value = self.value_conv(x).view(B, -1, H * W)
        out = torch.bmm(proj_value, energy).view(B, C, H, W)
        out = self.gamma(out)
        return out


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'input_channel_num': 4}]
