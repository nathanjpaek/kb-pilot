import torch
from copy import deepcopy
import torch.nn as nn
from functools import partial
from torch.nn.init import _calculate_correct_fan


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


def get_inputs():
    return [torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'in_features': 4, 'out_features': 4}]
