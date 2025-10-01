import torch
import torch.nn as nn
import torch.nn.functional as F


class Swish(nn.Module):

    def __init__(self):
        super(Swish, self).__init__()

    def forward(self, x):
        return x * x.sigmoid()


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


class SqueezeAndExcitationModule(nn.Module):
    """Squeeze And Excitation Module

    Args:
        input_dim: input feature dimension
        reduction_ratio: bottleneck reduction ratio
        inner_act: bottleneck inner activation function

    Input: (batch_size, in_dim, in_length)
    Output: (batch_size, out_dim, out_length)
    
    """

    def __init__(self, input_dim, reduction_ratio, inner_act='relu'):
        super(SqueezeAndExcitationModule, self).__init__()
        assert input_dim % reduction_ratio == 0
        self.conv1 = Conv1d(input_dim, input_dim // reduction_ratio,
            kernel_size=1)
        self.conv2 = Conv1d(input_dim // reduction_ratio, input_dim,
            kernel_size=1)
        assert inner_act in ['relu', 'swish']
        if inner_act == 'relu':
            self.inner_act = nn.ReLU()
        elif inner_act == 'swish':
            self.inner_act = Swish()

    def forward(self, x):
        scale = x.mean(dim=-1, keepdim=True)
        scale = self.conv1(scale)
        scale = self.inner_act(scale)
        scale = self.conv2(scale)
        scale = scale.sigmoid()
        x = x * scale
        return x


def get_inputs():
    return [torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'input_dim': 4, 'reduction_ratio': 4}]
