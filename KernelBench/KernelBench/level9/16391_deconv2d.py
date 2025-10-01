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


class deconv2d(nn.Module):

    def __init__(self, in_channels, out_channels, padding, kernel_size=(4, 
        4), stride=(2, 2), spectral_normed=False, iter=1):
        super(deconv2d, self).__init__()
        self.devconv = nn.ConvTranspose2d(in_channels, out_channels,
            kernel_size, stride, padding=padding)
        if spectral_normed:
            self.devconv = spectral_norm(self.deconv)

    def forward(self, input):
        out = self.devconv(input)
        return out


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 4, 'out_channels': 4, 'padding': 4}]
