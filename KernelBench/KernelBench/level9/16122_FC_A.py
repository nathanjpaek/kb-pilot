import math
import torch
import torch.nn as nn


def quick_scale(module, name='weight'):
    ScaleW.apply(module, name)
    return module


class ScaleW:
    """
    Constructor: name - name of attribute to be scaled
    """

    def __init__(self, name):
        self.name = name

    def scale(self, module):
        weight = getattr(module, self.name + '_orig')
        fan_in = weight.data.size(1) * weight.data[0][0].numel()
        return weight * math.sqrt(2 / fan_in)

    @staticmethod
    def apply(module, name):
        """
        Apply runtime scaling to specific module
        """
        hook = ScaleW(name)
        weight = getattr(module, name)
        module.register_parameter(name + '_orig', nn.Parameter(weight.data))
        del module._parameters[name]
        module.register_forward_pre_hook(hook)

    def __call__(self, module, whatever):
        weight = self.scale(module)
        setattr(module, self.name, weight)


class SLinear(nn.Module):

    def __init__(self, dim_in, dim_out):
        super().__init__()
        linear = nn.Linear(dim_in, dim_out)
        linear.weight.data.normal_()
        linear.bias.data.zero_()
        self.linear = quick_scale(linear)

    def forward(self, x):
        return self.linear(x)


class FC_A(nn.Module):
    """
    Learned affine transform A, this module is used to transform
    midiate vector w into a style vector
    """

    def __init__(self, dim_latent, n_channel):
        super().__init__()
        self.transform = SLinear(dim_latent, n_channel * 2)
        self.transform.linear.bias.data[:n_channel] = 1
        self.transform.linear.bias.data[n_channel:] = 0

    def forward(self, w):
        style = self.transform(w).unsqueeze(2).unsqueeze(3)
        return style


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'dim_latent': 4, 'n_channel': 4}]
