import torch
import torch.nn as nn
from torch.nn.init import _calculate_correct_fan
import torch.utils.cpp_extension


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


class AdaptiveInstanceNorm(nn.Module):
    """Adaptive Instance Normalization Module.

    Ref: https://github.com/rosinality/style-based-gan-pytorch/blob/master/model.py  # noqa

    Args:
        in_channel (int): The number of input's channel.
        style_dim (int): Style latent dimension.
    """

    def __init__(self, in_channel, style_dim):
        super().__init__()
        self.norm = nn.InstanceNorm2d(in_channel)
        self.affine = EqualizedLRLinearModule(style_dim, in_channel * 2)
        self.affine.bias.data[:in_channel] = 1
        self.affine.bias.data[in_channel:] = 0

    def forward(self, input, style):
        """Forward function.

        Args:
            input (Tensor): Input tensor with shape (n, c, h, w).
            style (Tensor): Input style tensor with shape (n, c).

        Returns:
            Tensor: Forward results.
        """
        style = self.affine(style).unsqueeze(2).unsqueeze(3)
        gamma, beta = style.chunk(2, 1)
        out = self.norm(input)
        out = gamma * out + beta
        return out


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'in_channel': 4, 'style_dim': 4}]
