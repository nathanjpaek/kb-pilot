import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import BatchNorm1d
from torch.nn import BatchNorm2d
from torch.nn import BatchNorm3d
from torch.nn import Identity
from torch.nn import GroupNorm
from torch.nn import InstanceNorm1d
from torch.nn import InstanceNorm2d
from torch.nn import InstanceNorm3d
from torch.nn import LayerNorm
from torch.nn import SyncBatchNorm


def BuildActivation(activation_type, **kwargs):
    supported_activations = {'relu': nn.ReLU, 'gelu': nn.GELU, 'relu6': nn.
        ReLU6, 'prelu': nn.PReLU, 'sigmoid': nn.Sigmoid, 'hardswish':
        HardSwish, 'identity': nn.Identity, 'leakyrelu': nn.LeakyReLU,
        'hardsigmoid': HardSigmoid}
    assert activation_type in supported_activations, 'unsupport activation type %s...' % activation_type
    return supported_activations[activation_type](**kwargs)


def BuildNormalization(norm_type='batchnorm2d', instanced_params=(0, {}),
    only_get_all_supported=False, **kwargs):
    supported_dict = {'identity': Identity, 'layernorm': LayerNorm,
        'groupnorm': GroupNorm, 'batchnorm1d': BatchNorm1d, 'batchnorm2d':
        BatchNorm2d, 'batchnorm3d': BatchNorm3d, 'syncbatchnorm':
        SyncBatchNorm, 'instancenorm1d': InstanceNorm1d, 'instancenorm2d':
        InstanceNorm2d, 'instancenorm3d': InstanceNorm3d}
    if only_get_all_supported:
        return list(supported_dict.values())
    assert norm_type in supported_dict, 'unsupport norm_type %s...' % norm_type
    norm_layer = supported_dict[norm_type](instanced_params[0], **
        instanced_params[1])
    return norm_layer


class HardSigmoid(nn.Module):

    def __init__(self, bias=1.0, divisor=2.0, min_value=0.0, max_value=1.0):
        super(HardSigmoid, self).__init__()
        assert divisor != 0, 'divisor is not allowed to be equal to zero'
        self.bias = bias
        self.divisor = divisor
        self.min_value = min_value
        self.max_value = max_value
    """forward"""

    def forward(self, x):
        x = (x + self.bias) / self.divisor
        return x.clamp_(self.min_value, self.max_value)


class HardSwish(nn.Module):

    def __init__(self, inplace=False):
        super(HardSwish, self).__init__()
        self.act = nn.ReLU6(inplace)
    """forward"""

    def forward(self, x):
        return x * self.act(x + 3) / 6


class AdptivePaddingConv2d(nn.Conv2d):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
        padding=0, dilation=1, groups=1, bias=True, norm_cfg=None, act_cfg=None
        ):
        super(AdptivePaddingConv2d, self).__init__(in_channels=in_channels,
            out_channels=out_channels, kernel_size=kernel_size, stride=
            stride, padding=0, dilation=dilation, groups=groups, bias=bias)
        if norm_cfg is not None:
            self.norm = BuildNormalization(norm_cfg['type'], (out_channels,
                norm_cfg['opts']))
        if act_cfg is not None:
            self.activation = BuildActivation(act_cfg['type'], **act_cfg[
                'opts'])
    """forward"""

    def forward(self, x):
        img_h, img_w = x.size()[-2:]
        kernel_h, kernel_w = self.weight.size()[-2:]
        stride_h, stride_w = self.stride
        output_h = math.ceil(img_h / stride_h)
        output_w = math.ceil(img_w / stride_w)
        pad_h = max((output_h - 1) * self.stride[0] + (kernel_h - 1) * self
            .dilation[0] + 1 - img_h, 0)
        pad_w = max((output_w - 1) * self.stride[1] + (kernel_w - 1) * self
            .dilation[1] + 1 - img_w, 0)
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h -
                pad_h // 2])
        output = F.conv2d(x, self.weight, self.bias, self.stride, self.
            padding, self.dilation, self.groups)
        if hasattr(self, 'norm'):
            output = self.norm(output)
        if hasattr(self, 'activation'):
            output = self.activation(output)
        return output


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}]
