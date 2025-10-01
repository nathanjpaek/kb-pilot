import torch
import torch.nn as nn
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


class AdaptiveInstanceNorm2d(nn.Module):

    def __init__(self, eps=1e-08):
        super(AdaptiveInstanceNorm2d, self).__init__()
        self.eps = eps

    def IN_noWeight(self, x):
        N, C = x.size(0), x.size(1)
        mean = x.contiguous().view(N, C, -1).mean(2).contiguous().view(N, C,
            1, 1)
        x = x - mean
        var = torch.mul(x, x)
        var = var.contiguous().view(N, C, -1).mean(2).contiguous().view(N,
            C, 1, 1)
        var = torch.rsqrt(var + self.eps)
        x = x * var
        return x

    def Apply_style(self, content, style):
        style = style.contiguous().view([-1, 2, content.size(1), 1, 1])
        content = content * style[:, 0] + style[:, 1]
        return content

    def forward(self, content, style):
        normalized_content = self.IN_noWeight(content)
        stylized_content = self.Apply_style(normalized_content, style)
        return stylized_content


class AdaINConv2dLayer(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
        padding=0, dilation=1, pad_type='zero', activation='lrelu', sn=True):
        super(AdaINConv2dLayer, self).__init__()
        if pad_type == 'reflect':
            self.pad = nn.ReflectionPad2d(padding)
        elif pad_type == 'replicate':
            self.pad = nn.ReplicationPad2d(padding)
        elif pad_type == 'zero':
            self.pad = nn.ZeroPad2d(padding)
        else:
            assert 0, 'Unsupported padding type: {}'.format(pad_type)
        self.norm = AdaptiveInstanceNorm2d()
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU(inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, 'Unsupported activation: {}'.format(activation)
        if sn:
            self.conv2d = SpectralNorm(nn.Conv2d(in_channels, out_channels,
                kernel_size, stride, padding=0, dilation=dilation))
        else:
            self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size,
                stride, padding=0, dilation=dilation)

    def forward(self, x, style):
        x = self.pad(x)
        x = self.conv2d(x)
        x = self.norm(x, style)
        if self.activation:
            x = self.activation(x)
        return x


class TransposeAdaINConv2dLayer(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
        padding=0, dilation=1, pad_type='zero', activation='lrelu', sn=True,
        scale_factor=2):
        super(TransposeAdaINConv2dLayer, self).__init__()
        self.scale_factor = scale_factor
        self.conv2d = AdaINConv2dLayer(in_channels, out_channels,
            kernel_size, stride, padding, dilation, pad_type, activation, sn)

    def forward(self, x, style):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode='nearest')
        x = self.conv2d(x, style)
        return x


def get_inputs():
    return [torch.rand([256, 4, 4, 4]), torch.rand([32, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}]
