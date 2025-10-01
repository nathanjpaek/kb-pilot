import torch
import torch.nn.functional as F
from copy import deepcopy
import torch.nn as nn
from functools import partial
from torch.nn.init import _calculate_correct_fan


def upsample(in_tens, out_H=64):
    """Upsamples the input to the given size.

    Args:
        in_tens (Tensor): Tensor with shape [N, C, H, W].
        out_H (int, optional): Output spatial size. Defaults to 64.

    Returns:
        Tensor: Output Tensor.
    """
    in_H = in_tens.shape[2]
    scale_factor = 1.0 * out_H / in_H
    return nn.Upsample(scale_factor=scale_factor, mode='bilinear',
        align_corners=False)(in_tens)


def equalized_lr(module, name='weight', gain=2 ** 0.5, mode='fan_in',
    lr_mul=1.0):
    """Equalized Learning Rate.

    This trick is proposed in:
    Progressive Growing of GANs for Improved Quality, Stability, and Variation

    The general idea is to dynamically rescale the weight in training instead
    of in initializing so that the variance of the responses in each layer is
    guaranteed with some statistical properties.

    Note that this function is always combined with a convolution module which
    is initialized with :math:`\\mathcal{N}(0, 1)`.

    Args:
        module (nn.Module): Module to be wrapped.
        name (str | optional): The name of weights. Defaults to 'weight'.
        mode (str, optional): The mode of computing ``fan`` which is the
            same as ``kaiming_init`` in pytorch. You can choose one from
            ['fan_in', 'fan_out']. Defaults to 'fan_in'.

    Returns:
        nn.Module: Module that is registered with equalized lr hook.
    """
    EqualizedLR.apply(module, name, gain=gain, mode=mode, lr_mul=lr_mul)
    return module


def _make_kernel(k):
    k = torch.tensor(k, dtype=torch.float32)
    if k.ndim == 1:
        k = k[None, :] * k[:, None]
    k /= k.sum()
    return k


class EqualizedLR:
    """Equalized Learning Rate.

    This trick is proposed in:
    Progressive Growing of GANs for Improved Quality, Stability, and Variation

    The general idea is to dynamically rescale the weight in training instead
    of in initializing so that the variance of the responses in each layer is
    guaranteed with some statistical properties.

    Note that this function is always combined with a convolution module which
    is initialized with :math:`\\mathcal{N}(0, 1)`.

    Args:
        name (str | optional): The name of weights. Defaults to 'weight'.
        mode (str, optional): The mode of computing ``fan`` which is the
            same as ``kaiming_init`` in pytorch. You can choose one from
            ['fan_in', 'fan_out']. Defaults to 'fan_in'.
    """

    def __init__(self, name='weight', gain=2 ** 0.5, mode='fan_in', lr_mul=1.0
        ):
        self.name = name
        self.mode = mode
        self.gain = gain
        self.lr_mul = lr_mul

    def compute_weight(self, module):
        """Compute weight with equalized learning rate.

        Args:
            module (nn.Module): A module that is wrapped with equalized lr.

        Returns:
            torch.Tensor: Updated weight.
        """
        weight = getattr(module, self.name + '_orig')
        if weight.ndim == 5:
            fan = _calculate_correct_fan(weight[0], self.mode)
        else:
            assert weight.ndim <= 4
            fan = _calculate_correct_fan(weight, self.mode)
        weight = weight * torch.tensor(self.gain, device=weight.device
            ) * torch.sqrt(torch.tensor(1.0 / fan, device=weight.device)
            ) * self.lr_mul
        return weight

    def __call__(self, module, inputs):
        """Standard interface for forward pre hooks."""
        setattr(module, self.name, self.compute_weight(module))

    @staticmethod
    def apply(module, name, gain=2 ** 0.5, mode='fan_in', lr_mul=1.0):
        """Apply function.

        This function is to register an equalized learning rate hook in an
        ``nn.Module``.

        Args:
            module (nn.Module): Module to be wrapped.
            name (str | optional): The name of weights. Defaults to 'weight'.
            mode (str, optional): The mode of computing ``fan`` which is the
                same as ``kaiming_init`` in pytorch. You can choose one from
                ['fan_in', 'fan_out']. Defaults to 'fan_in'.

        Returns:
            nn.Module: Module that is registered with equalized lr hook.
        """
        for _, hook in module._forward_pre_hooks.items():
            if isinstance(hook, EqualizedLR):
                raise RuntimeError(
                    f'Cannot register two equalized_lr hooks on the same parameter {name} in {module} module.'
                    )
        fn = EqualizedLR(name, gain=gain, mode=mode, lr_mul=lr_mul)
        weight = module._parameters[name]
        delattr(module, name)
        module.register_parameter(name + '_orig', weight)
        setattr(module, name, weight.data)
        module.register_forward_pre_hook(fn)
        return fn


class EqualizedLRLinearModule(nn.Linear):
    """Equalized LR LinearModule.

    In this module, we adopt equalized lr in ``nn.Linear``. The equalized
    learning rate is proposed in:
    Progressive Growing of GANs for Improved Quality, Stability, and Variation

    Note that, the initialization of ``self.weight`` will be overwritten as
    :math:`\\mathcal{N}(0, 1)`.

    Args:
        equalized_lr_cfg (dict | None, optional): Config for ``EqualizedLR``.
            If ``None``, equalized learning rate is ignored. Defaults to
            dict(mode='fan_in').
    """

    def __init__(self, *args, equalized_lr_cfg=dict(mode='fan_in'), **kwargs):
        super().__init__(*args, **kwargs)
        self.with_equalized_lr = equalized_lr_cfg is not None
        if self.with_equalized_lr:
            self.lr_mul = equalized_lr_cfg.get('lr_mul', 1.0)
        else:
            self.lr_mul = 1.0
        if self.with_equalized_lr:
            equalized_lr(self, **equalized_lr_cfg)
            self._init_linear_weights()

    def _init_linear_weights(self):
        """Initialize linear weights as described in PGGAN."""
        nn.init.normal_(self.weight, 0, 1.0 / self.lr_mul)
        if self.bias is not None:
            nn.init.constant_(self.bias, 0.0)


class EqualLinearActModule(nn.Module):
    """Equalized LR Linear Module with Activation Layer.

    This module is modified from ``EqualizedLRLinearModule`` defined in PGGAN.
    The major features updated in this module is adding support for activation
    layers used in StyleGAN2.

    Args:
        equalized_lr_cfg (dict | None, optional): Config for equalized lr.
            Defaults to dict(gain=1., lr_mul=1.).
        bias (bool, optional): Whether to use bias item. Defaults to True.
        bias_init (float, optional): The value for bias initialization.
            Defaults to ``0.``.
        act_cfg (dict | None, optional): Config for activation layer.
            Defaults to None.
    """

    def __init__(self, *args, equalized_lr_cfg=dict(gain=1.0, lr_mul=1.0),
        bias=True, bias_init=0.0, act_cfg=None, **kwargs):
        super().__init__()
        self.with_activation = act_cfg is not None
        self.linear = EqualizedLRLinearModule(*args, bias=False,
            equalized_lr_cfg=equalized_lr_cfg, **kwargs)
        if equalized_lr_cfg is not None:
            self.lr_mul = equalized_lr_cfg.get('lr_mul', 1.0)
        else:
            self.lr_mul = 1.0
        if bias:
            self.bias = nn.Parameter(torch.zeros(self.linear.out_features).
                fill_(bias_init))
        else:
            self.bias = None
        if self.with_activation:
            act_cfg = deepcopy(act_cfg)
            if act_cfg['type'] == 'fused_bias':
                self.act_type = act_cfg.pop('type')
                assert self.bias is not None
                self.activate = partial(fused_bias_leakyrelu, **act_cfg)
            else:
                self.act_type = 'normal'
                self.activate = build_activation_layer(act_cfg)
        else:
            self.act_type = None

    def forward(self, x):
        """Forward function.

        Args:
            x (Tensor): Input feature map with shape of (N, C, ...).

        Returns:
            Tensor: Output feature map.
        """
        if x.ndim >= 3:
            x = x.reshape(x.size(0), -1)
        x = self.linear(x)
        if self.with_activation and self.act_type == 'fused_bias':
            x = self.activate(x, self.bias * self.lr_mul)
        elif self.bias is not None and self.with_activation:
            x = self.activate(x + self.bias * self.lr_mul)
        elif self.bias is not None:
            x = x + self.bias * self.lr_mul
        elif self.with_activation:
            x = self.activate(x)
        return x


class Blur(nn.Module):
    """Blur module.

    This module is adopted rightly after upsampling operation in StyleGAN2.

    Args:
        kernel (Array): Blur kernel/filter used in UpFIRDn.
        pad (list[int]): Padding for features.
        upsample_factor (int, optional): Upsampling factor. Defaults to 1.
    """

    def __init__(self, kernel, pad, upsample_factor=1):
        super().__init__()
        kernel = _make_kernel(kernel)
        if upsample_factor > 1:
            kernel = kernel * upsample_factor ** 2
        self.register_buffer('kernel', kernel)
        self.pad = pad

    def forward(self, x):
        """Forward function.

        Args:
            x (Tensor): Input feature map with shape of (N, C, H, W).

        Returns:
            Tensor: Output feature map.
        """
        return upfirdn2d(x, self.kernel, pad=self.pad)


class ModulatedConv2d(nn.Module):
    """Modulated Conv2d in StyleGANv2.

    This module implements the modulated convolution layers proposed in
    StyleGAN2. Details can be found in Analyzing and Improving the Image
    Quality of StyleGAN, CVPR2020.

    Args:
        in_channels (int): Input channels.
        out_channels (int): Output channels.
        kernel_size (int): Kernel size, same as :obj:`nn.Con2d`.
        style_channels (int): Channels for the style codes.
        demodulate (bool, optional): Whether to adopt demodulation.
            Defaults to True.
        upsample (bool, optional): Whether to adopt upsampling in features.
            Defaults to False.
        downsample (bool, optional): Whether to adopt downsampling in features.
            Defaults to False.
        blur_kernel (list[int], optional): Blurry kernel.
            Defaults to [1, 3, 3, 1].
        equalized_lr_cfg (dict | None, optional): Configs for equalized lr.
            Defaults to dict(mode='fan_in', lr_mul=1., gain=1.).
        style_mod_cfg (dict, optional): Configs for style modulation module.
            Defaults to dict(bias_init=1.).
        style_bias (float, optional): Bias value for style code.
            Defaults to 0..
        eps (float, optional): Epsilon value to avoid computation error.
            Defaults to 1e-8.
    """

    def __init__(self, in_channels, out_channels, kernel_size,
        style_channels, demodulate=True, upsample=False, downsample=False,
        blur_kernel=[1, 3, 3, 1], equalized_lr_cfg=dict(mode='fan_in',
        lr_mul=1.0, gain=1.0), style_mod_cfg=dict(bias_init=1.0),
        style_bias=0.0, eps=1e-08):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.style_channels = style_channels
        self.demodulate = demodulate
        assert isinstance(self.kernel_size, int) and (self.kernel_size >= 1 and
            self.kernel_size % 2 == 1)
        self.upsample = upsample
        self.downsample = downsample
        self.style_bias = style_bias
        self.eps = eps
        style_mod_cfg = dict() if style_mod_cfg is None else style_mod_cfg
        self.style_modulation = EqualLinearActModule(style_channels,
            in_channels, **style_mod_cfg)
        lr_mul_ = 1.0
        if equalized_lr_cfg is not None:
            lr_mul_ = equalized_lr_cfg.get('lr_mul', 1.0)
        self.weight = nn.Parameter(torch.randn(1, out_channels, in_channels,
            kernel_size, kernel_size).div_(lr_mul_))
        if upsample:
            factor = 2
            p = len(blur_kernel) - factor - (kernel_size - 1)
            pad0 = (p + 1) // 2 + factor - 1
            pad1 = p // 2 + 1
            self.blur = Blur(blur_kernel, (pad0, pad1), upsample_factor=factor)
        if downsample:
            factor = 2
            p = len(blur_kernel) - factor + (kernel_size - 1)
            pad0 = (p + 1) // 2
            pad1 = p // 2
            self.blur = Blur(blur_kernel, pad=(pad0, pad1))
        if equalized_lr_cfg is not None:
            equalized_lr(self, **equalized_lr_cfg)
        self.padding = kernel_size // 2

    def forward(self, x, style):
        n, c, h, w = x.shape
        style = self.style_modulation(style).view(n, 1, c, 1, 1
            ) + self.style_bias
        weight = self.weight * style
        if self.demodulate:
            demod = torch.rsqrt(weight.pow(2).sum([2, 3, 4]) + self.eps)
            weight = weight * demod.view(n, self.out_channels, 1, 1, 1)
        weight = weight.view(n * self.out_channels, c, self.kernel_size,
            self.kernel_size)
        if self.upsample:
            x = x.reshape(1, n * c, h, w)
            weight = weight.view(n, self.out_channels, c, self.kernel_size,
                self.kernel_size)
            weight = weight.transpose(1, 2).reshape(n * c, self.
                out_channels, self.kernel_size, self.kernel_size)
            x = F.conv_transpose2d(x, weight, padding=0, stride=2, groups=n)
            x = x.reshape(n, self.out_channels, *x.shape[-2:])
            x = self.blur(x)
        elif self.downsample:
            x = self.blur(x)
            x = x.view(1, n * self.in_channels, *x.shape[-2:])
            x = F.conv2d(x, weight, stride=2, padding=0, groups=n)
            x = x.view(n, self.out_channels, *x.shape[-2:])
        else:
            x = x.view(1, n * c, h, w)
            x = F.conv2d(x, weight, stride=1, padding=self.padding, groups=n)
            x = x.view(n, self.out_channels, *x.shape[-2:])
        return x


class UpsampleUpFIRDn(nn.Module):
    """UpFIRDn for Upsampling.

    This module is used in the ``to_rgb`` layers in StyleGAN2 for upsampling
    the images.

    Args:
        kernel (Array): Blur kernel/filter used in UpFIRDn.
        factor (int, optional): Upsampling factor. Defaults to 2.
    """

    def __init__(self, kernel, factor=2):
        super().__init__()
        self.factor = factor
        kernel = _make_kernel(kernel) * factor ** 2
        self.register_buffer('kernel', kernel)
        p = kernel.shape[0] - factor
        pad0 = (p + 1) // 2 + factor - 1
        pad1 = p // 2
        self.pad = pad0, pad1

    def forward(self, x):
        """Forward function.

        Args:
            x (Tensor): Input feature map with shape of (N, C, H, W).

        Returns:
            Tensor: Output feature map.
        """
        out = upfirdn2d(x, self.kernel, up=self.factor, down=1, pad=self.pad)
        return out


class ModulatedToRGB(nn.Module):
    """To RGB layer.

    This module is designed to output image tensor in StyleGAN2.

    Args:
        in_channels (int): Input channels.
        style_channels (int): Channels for the style codes.
        out_channels (int, optional): Output channels. Defaults to 3.
        upsample (bool, optional): Whether to adopt upsampling in features.
            Defaults to False.
        blur_kernel (list[int], optional): Blurry kernel.
            Defaults to [1, 3, 3, 1].
        style_mod_cfg (dict, optional): Configs for style modulation module.
            Defaults to dict(bias_init=1.).
        style_bias (float, optional): Bias value for style code.
            Defaults to 0..
    """

    def __init__(self, in_channels, style_channels, out_channels=3,
        upsample=True, blur_kernel=[1, 3, 3, 1], style_mod_cfg=dict(
        bias_init=1.0), style_bias=0.0):
        super().__init__()
        if upsample:
            self.upsample = UpsampleUpFIRDn(blur_kernel)
        self.conv = ModulatedConv2d(in_channels, out_channels=out_channels,
            kernel_size=1, style_channels=style_channels, demodulate=False,
            style_mod_cfg=style_mod_cfg, style_bias=style_bias)
        self.bias = nn.Parameter(torch.zeros(1, 3, 1, 1))

    def forward(self, x, style, skip=None):
        """Forward Function.

        Args:
            x ([Tensor): Input features with shape of (N, C, H, W).
            style (Tensor): Style latent with shape of (N, C).
            skip (Tensor, optional): Tensor for skip link. Defaults to None.

        Returns:
            Tensor: Output features with shape of (N, C, H, W)
        """
        out = self.conv(x, style)
        out = out + self.bias
        if skip is not None:
            skip = self.upsample(skip)
            out = out + skip
        return out


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 4, 'style_channels': 4}]
