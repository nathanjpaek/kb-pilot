import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv1d(nn.Conv1d):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
        padding='same', dilation=1, groups=1, bias=True):
        super(Conv1d, self).__init__(in_channels=in_channels, out_channels=
            out_channels, kernel_size=kernel_size, stride=stride, padding=0,
            dilation=dilation, groups=groups, bias=bias, padding_mode='zeros')
        assert padding in ['valid', 'same', 'causal']
        if padding == 'valid':
            self.pre_padding = None
        elif padding == 'same':
            self.pre_padding = nn.ConstantPad1d(padding=((kernel_size - 1) //
                2, (kernel_size - 1) // 2), value=0)
        elif padding == 'causal':
            self.pre_padding = nn.ConstantPad1d(padding=(kernel_size - 1, 0
                ), value=0)
        self.noise = None
        self.vn_std = None

    def init_vn(self, vn_std):
        self.vn_std = vn_std

    def sample_synaptic_noise(self, distributed):
        self.noise = torch.normal(mean=0.0, std=1.0, size=self.weight.size(
            ), device=self.weight.device, dtype=self.weight.dtype)
        if distributed:
            torch.distributed.broadcast(self.noise, 0)

    def forward(self, input):
        weight = self.weight
        if self.noise is not None and self.training:
            weight = weight + self.vn_std * self.noise
        if self.pre_padding is not None:
            input = self.pre_padding(input)
        return F.conv1d(input, weight, self.bias, self.stride, self.padding,
            self.dilation, self.groups)


def get_inputs():
    return [torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}]
