import torch
import torch.nn.functional as F
import torch.nn as nn


def l2normalize(v, esp=1e-08):
    return v / (v.norm() + esp)


def sn_weight(weight, u, height, n_power_iterations):
    weight.requires_grad_(False)
    for _ in range(n_power_iterations):
        v = l2normalize(torch.mv(weight.view(height, -1).t(), u))
        u = l2normalize(torch.mv(weight.view(height, -1), v))
    weight.requires_grad_(True)
    sigma = u.dot(weight.view(height, -1).mv(v))
    return torch.div(weight, sigma), u


def get_nonlinearity(nonlinearity=None):
    if not nonlinearity:
        pass
    elif callable(nonlinearity):
        if nonlinearity == nn.LeakyReLU:
            nonlinearity = nonlinearity(0.02, inplace=True)
    elif hasattr(nn, nonlinearity):
        nonlinearity = getattr(nn, nonlinearity)
        if nonlinearity == 'LeakyReLU':
            nonlinearity = nonlinearity(0.02, inplace=True)
        else:
            nonlinearity = nonlinearity()
    elif hasattr(nn.functional, nonlinearity):
        nonlinearity = getattr(nn.functional, nonlinearity)
    else:
        raise ValueError(nonlinearity)
    return nonlinearity


class SNConv2d(nn.Conv2d):

    def __init__(self, *args, n_power_iterations=1, **kwargs):
        super(SNConv2d, self).__init__(*args, **kwargs)
        self.n_power_iterations = n_power_iterations
        self.height = self.weight.shape[0]
        self.register_buffer('u', l2normalize(self.weight.new_empty(self.
            height).normal_(0, 1)))

    def forward(self, input):
        w_sn, self.u = sn_weight(self.weight, self.u, self.height, self.
            n_power_iterations)
        return F.conv2d(input, w_sn, self.bias, self.stride, self.padding,
            self.dilation, self.groups)


class ConvMeanPool(nn.Module):

    def __init__(self, dim_in, dim_out, f_size, nonlinearity=None, prefix=
        '', spectral_norm=False):
        super(ConvMeanPool, self).__init__()
        Conv2d = SNConv2d if spectral_norm else nn.Conv2d
        models = nn.Sequential()
        nonlinearity = get_nonlinearity(nonlinearity)
        name = 'cmp' + prefix
        models.add_module(name, Conv2d(dim_in, dim_out, f_size, 1, 1, bias=
            False))
        models.add_module(name + '_pool', nn.AvgPool2d(2, count_include_pad
            =False))
        if nonlinearity:
            models.add_module('{}_{}'.format(name, nonlinearity.__class__.
                __name__), nonlinearity)
        self.models = models

    def forward(self, x):
        x = self.models(x)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'dim_in': 4, 'dim_out': 4, 'f_size': 4}]
