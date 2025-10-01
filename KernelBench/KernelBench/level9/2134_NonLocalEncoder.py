import torch
from torch import Tensor
from torch import nn
import torch.utils.data
from typing import Optional
from abc import ABCMeta


class _NonLocalNd(nn.Module, metaclass=ABCMeta):
    """Basic Non-local module.

    This module is proposed in
    "Non-local Neural Networks"
    Paper reference: https://arxiv.org/abs/1711.07971
    Code reference: https://github.com/AlexHex7/Non-local_pytorch

    Args:
        in_channels (int): Channels of the input feature map.
        reduction (int): Channel reduction ratio. Default: 2.
        use_scale (bool): Whether to scale pairwise_weight by
            `1/sqrt(inter_channels)` when the mode is `embedded_gaussian`.
            Default: True.
        conv_cfg (None | dict): The config dict for convolution layers.
            If not specified, it will use `nn.Conv2d` for convolution layers.
            Default: None.
        norm_cfg (None | dict): The config dict for normalization layers.
            Default: None. (This parameter is only applicable to conv_out.)
        mode (str): Options are `gaussian`, `concatenation`,
            `embedded_gaussian` and `dot_product`. Default: embedded_gaussian.
    """

    def __init__(self, in_channels, reduction=2, use_scale=True, conv_cfg=
        None, norm_cfg=None, mode='embedded_gaussian', **kwargs):
        super(_NonLocalNd, self).__init__()
        self.in_channels = in_channels
        self.reduction = reduction
        self.use_scale = use_scale
        self.inter_channels = max(in_channels // reduction, 1)
        self.mode = mode
        if mode not in ['gaussian', 'embedded_gaussian', 'dot_product',
            'concatenation']:
            raise ValueError(
                f"Mode should be in 'gaussian', 'concatenation', 'embedded_gaussian' or 'dot_product', but got {mode} instead."
                )
        self.g = nn.Conv2d(self.in_channels, self.inter_channels, 1)
        self.conv_out = nn.Conv2d(self.inter_channels, self.in_channels, 1)
        if self.mode != 'gaussian':
            self.theta = nn.Conv2d(self.in_channels, self.inter_channels, 1)
            self.phi = nn.Conv2d(self.in_channels, self.inter_channels, 1)
        if self.mode == 'concatenation':
            self.concat_project = nn.Sequential(nn.Conv2d(self.
                inter_channels * 2, 1, kernel_size=1, stride=1, padding=0,
                bias=False), nn.ReLU())

    def gaussian(self, theta_x, phi_x):
        pairwise_weight = torch.matmul(theta_x, phi_x)
        pairwise_weight = pairwise_weight.softmax(dim=-1)
        return pairwise_weight

    def embedded_gaussian(self, theta_x, phi_x):
        pairwise_weight = torch.matmul(theta_x, phi_x)
        if self.use_scale:
            pairwise_weight /= theta_x.shape[-1] ** 0.5
        pairwise_weight = pairwise_weight.softmax(dim=-1)
        return pairwise_weight

    def dot_product(self, theta_x, phi_x):
        pairwise_weight = torch.matmul(theta_x, phi_x)
        pairwise_weight /= pairwise_weight.shape[-1]
        return pairwise_weight

    def concatenation(self, theta_x, phi_x):
        h = theta_x.size(2)
        w = phi_x.size(3)
        theta_x = theta_x.repeat(1, 1, 1, w)
        phi_x = phi_x.repeat(1, 1, h, 1)
        concat_feature = torch.cat([theta_x, phi_x], dim=1)
        pairwise_weight = self.concat_project(concat_feature)
        n, _, h, w = pairwise_weight.size()
        pairwise_weight = pairwise_weight.view(n, h, w)
        pairwise_weight /= pairwise_weight.shape[-1]
        return pairwise_weight

    def forward(self, x):
        n = x.size(0)
        g_x = self.g(x).view(n, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)
        if self.mode == 'gaussian':
            theta_x = x.view(n, self.in_channels, -1)
            theta_x = theta_x.permute(0, 2, 1)
            if self.sub_sample:
                phi_x = self.phi(x).view(n, self.in_channels, -1)
            else:
                phi_x = x.view(n, self.in_channels, -1)
        elif self.mode == 'concatenation':
            theta_x = self.theta(x).view(n, self.inter_channels, -1, 1)
            phi_x = self.phi(x).view(n, self.inter_channels, 1, -1)
        else:
            theta_x = self.theta(x).view(n, self.inter_channels, -1)
            theta_x = theta_x.permute(0, 2, 1)
            phi_x = self.phi(x).view(n, self.inter_channels, -1)
        pairwise_func = getattr(self, self.mode)
        pairwise_weight = pairwise_func(theta_x, phi_x)
        y = torch.matmul(pairwise_weight, g_x)
        y = y.permute(0, 2, 1).contiguous().reshape(n, self.inter_channels,
            *x.size()[2:])
        output = x + self.conv_out(y)
        return output


class NonLocal2d(_NonLocalNd):
    """2D Non-local module.

    Args:
        in_channels (int): Same as `NonLocalND`.
        sub_sample (bool): Whether to apply max pooling after pairwise
            function (Note that the `sub_sample` is applied on spatial only).
            Default: False.
        conv_cfg (None | dict): Same as `NonLocalND`.
            Default: dict(type='Conv2d').
    """
    _abbr_ = 'nonlocal_block'

    def __init__(self, in_channels, sub_sample=False, conv_cfg=dict(type=
        'Conv2d'), **kwargs):
        super(NonLocal2d, self).__init__(in_channels, conv_cfg=conv_cfg, **
            kwargs)
        self.sub_sample = sub_sample
        if sub_sample:
            max_pool_layer = nn.MaxPool2d(kernel_size=(2, 2))
            self.g = nn.Sequential(self.g, max_pool_layer)
            if self.mode != 'gaussian':
                self.phi = nn.Sequential(self.phi, max_pool_layer)
            else:
                self.phi = max_pool_layer


class NonLocalEncoder(nn.Module):

    def __init__(self, d_model, norm=None):
        super().__init__()
        self.layer = NonLocal2d(d_model)
        self.norm = norm

    def forward(self, src, mask: 'Optional[Tensor]'=None,
        src_key_padding_mask: 'Optional[Tensor]'=None, pos:
        'Optional[Tensor]'=None):
        output = src
        output = self.layer(src)
        output = output.flatten(2).permute(2, 0, 1)
        if self.norm is not None:
            output = self.norm(output)
        return output


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'d_model': 4}]
