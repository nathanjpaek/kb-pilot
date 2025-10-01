import torch
import torch.nn as nn
from torch.autograd import Variable


def spectral_norm(module, name='weight'):
    SpectralNorm.apply(module, name)
    return module


class SpectralNorm:

    def __init__(self, name):
        self.name = name

    def compute_weight(self, module):
        weight = getattr(module, self.name + '_orig')
        u = getattr(module, self.name + '_u')
        size = weight.size()
        weight_mat = weight.contiguous().view(size[0], -1)
        if weight_mat.is_cuda:
            u = u
        v = weight_mat.t() @ u
        v = v / v.norm()
        u = weight_mat @ v
        u = u / u.norm()
        weight_sn = weight_mat / (u.t() @ weight_mat @ v)
        weight_sn = weight_sn.view(*size)
        return weight_sn, Variable(u.data)

    @staticmethod
    def apply(module, name):
        fn = SpectralNorm(name)
        weight = getattr(module, name)
        del module._parameters[name]
        module.register_parameter(name + '_orig', nn.Parameter(weight.data))
        input_size = weight.size(0)
        u = Variable(torch.randn(input_size, 1) * 0.1, requires_grad=False)
        setattr(module, name + '_u', u)
        setattr(module, name, fn.compute_weight(module)[0])
        module.register_forward_pre_hook(fn)
        return fn

    def __call__(self, module, input):
        weight_sn, u = self.compute_weight(module)
        setattr(module, self.name, weight_sn)
        setattr(module, self.name + '_u', u)


class conv2d(nn.Module):

    def __init__(self, in_channels, out_channels, padding, kernel_size=4,
        stride=2, spectral_normed=False):
        super(conv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
            stride, padding=padding)
        if spectral_normed:
            self.conv = spectral_norm(self.conv)

    def forward(self, input):
        out = self.conv(input)
        return out


class Residual_D(nn.Module):

    def __init__(self, in_channels, out_channels, kernel=3, stride=1,
        spectral_normed=False, down_sampling=False, is_start=False):
        super(Residual_D, self).__init__()
        self.down_sampling = down_sampling
        self.is_start = is_start
        self.avgpool_short = nn.AvgPool2d(2, 2, padding=1)
        self.conv_short = conv2d(in_channels, out_channels, kernel_size=1,
            stride=1, padding=0, spectral_normed=False)
        self.conv1 = conv2d(in_channels, out_channels, spectral_normed=
            spectral_normed, kernel_size=kernel, stride=stride, padding=1)
        self.conv2 = conv2d(out_channels, out_channels, spectral_normed=
            spectral_normed, kernel_size=kernel, stride=stride, padding=1)
        self.avgpool2 = nn.AvgPool2d(2, 2, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        input = x
        if self.is_start:
            conv1 = self.relu(self.conv1(x))
            conv2 = self.relu(self.conv2(conv1))
            if self.down_sampling:
                conv2 = self.avgpool2(conv2)
        else:
            conv1 = self.conv1(self.relu(x))
            conv2 = self.conv2(self.relu(conv1))
            if self.down_sampling:
                conv2 = self.avgpool2(conv2)
        if self.down_sampling:
            input = self.avgpool_short(input)
        resi = self.conv_short(input)
        return resi + conv2


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 4, 'out_channels': 4}]
