import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt


def equal_lr(module, name='weight'):
    EqualLR.apply(module, name)
    return module


class EqualLR:

    def __init__(self, name):
        self.name = name

    def compute_weight(self, module):
        weight = getattr(module, self.name + '_orig')
        fan_in = weight.data.size(1) * weight.data[0][0].numel()
        return weight * sqrt(2 / fan_in)

    @staticmethod
    def apply(module, name):
        fn = EqualLR(name)
        weight = getattr(module, name)
        del module._parameters[name]
        module.register_parameter(name + '_orig', nn.Parameter(weight.data))
        module.register_forward_pre_hook(fn)
        return fn

    def __call__(self, module, input):
        weight = self.compute_weight(module)
        setattr(module, self.name, weight)


class EqualLinear(nn.Module):

    def __init__(self, in_dim, out_dim):
        super().__init__()
        linear = nn.Linear(in_dim, out_dim)
        linear.weight.data.normal_()
        linear.bias.data.zero_()
        self.linear = equal_lr(linear)

    def forward(self, input):
        return self.linear(input)


class AdaptiveAttention(nn.Module):

    def __init__(self, img_dim, style_dim):
        super().__init__()
        self.img_dim = img_dim
        self.fc = EqualLinear(style_dim, img_dim ** 2)
        self.gamma = nn.Parameter(torch.ones(1, 1, 1, 1))

    def forward(self, x, p):
        h = self.fc(p)
        h = h.view(h.size(0), 1, self.img_dim, self.img_dim)
        h = F.sigmoid(h)
        return self.gamma * (h * x) + x


def get_inputs():
    return [torch.rand([4, 4]), torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'img_dim': 4, 'style_dim': 4}]
