import torch
import torch.utils.data
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import Parameter


def l2normalize(v, eps=1e-12):
    return v / (v.norm() + eps)


class SpectralNorm(nn.Module):

    def __init__(self, module, name='weight', power_iterations=1):
        super(SpectralNorm, self).__init__()
        self.module = module
        self.name = name
        self.power_iterations = power_iterations
        if not self._made_params():
            self._make_params()

    def _update_u_v(self):
        u = getattr(self.module, self.name + '_u')
        v = getattr(self.module, self.name + '_v')
        w = getattr(self.module, self.name + '_bar')
        height = w.data.shape[0]
        for _ in range(self.power_iterations):
            v.data = l2normalize(torch.mv(torch.t(w.view(height, -1).data),
                u.data))
            u.data = l2normalize(torch.mv(w.view(height, -1).data, v.data))
        sigma = u.dot(w.view(height, -1).mv(v))
        setattr(self.module, self.name, w / sigma.expand_as(w))

    def _made_params(self):
        try:
            getattr(self.module, self.name + '_u')
            getattr(self.module, self.name + '_v')
            getattr(self.module, self.name + '_bar')
            return True
        except AttributeError:
            return False

    def _make_params(self):
        w = getattr(self.module, self.name)
        height = w.data.shape[0]
        width = w.view(height, -1).data.shape[1]
        u = Parameter(w.data.new(height).normal_(0, 1), requires_grad=False)
        v = Parameter(w.data.new(width).normal_(0, 1), requires_grad=False)
        u.data = l2normalize(u.data)
        v.data = l2normalize(v.data)
        w_bar = Parameter(w.data)
        del self.module._parameters[self.name]
        self.module.register_parameter(self.name + '_u', u)
        self.module.register_parameter(self.name + '_v', v)
        self.module.register_parameter(self.name + '_bar', w_bar)

    def forward(self, *args):
        self._update_u_v()
        return self.module.forward(*args)


class _final_conv_block(nn.Module):

    def __init__(self, in_features, out_features, filter_size=3, stride=1,
        padding_mode='reflect', sn=False):
        super(_final_conv_block, self).__init__()
        self.pad_mode = padding_mode
        self.filter_size = filter_size
        self.conv1 = nn.Conv2d(in_features, out_features, filter_size,
            stride=stride)
        if sn:
            self.conv1 = SpectralNorm(self.conv1)

    def forward(self, x):
        n_pad_pxl = self.filter_size // 2
        n_pad_by_sides = n_pad_pxl, n_pad_pxl, n_pad_pxl, n_pad_pxl
        output = self.conv1(F.pad(x, n_pad_by_sides, mode=self.pad_mode))
        return output


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_features': 4, 'out_features': 4}]
