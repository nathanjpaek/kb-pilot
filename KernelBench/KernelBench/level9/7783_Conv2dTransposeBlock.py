import torch
from torch.nn import functional as F
from torch import nn
from torch.nn.utils import spectral_norm


class AdaptiveInstanceNorm2d(nn.Module):

    def __init__(self, num_features, eps=1e-05, momentum=0.1):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))

    def forward(self, x, adain_params):
        b, c = x.size(0), x.size(1)
        running_mean = self.running_mean.repeat(b)
        running_var = self.running_var.repeat(b)
        x_reshaped = x.contiguous().view(1, b * c, *x.size()[2:])
        out = F.batch_norm(x_reshaped, running_mean, running_var,
            adain_params['weight'], adain_params['bias'], True, self.
            momentum, self.eps)
        return out.view(b, c, *x.size()[2:])

    def __repr__(self):
        return self.__class__.__name__ + '(' + str(self.num_features) + ')'


class Conv2dTransposeBlock(nn.Module):

    def __init__(self, in_dim, out_dim, ks, st, padding=0, norm='none',
        activation='elu', use_bias=True, activation_first=False, snorm=False):
        super().__init__()
        self.use_bias = use_bias
        self.activation_first = activation_first
        norm_dim = out_dim
        if norm == 'bn':
            self.norm = nn.BatchNorm2d(norm_dim)
        elif norm == 'in':
            self.norm = nn.InstanceNorm2d(norm_dim)
        elif norm == 'group':
            self.norm = nn.GroupNorm(num_channels=norm_dim, num_groups=16)
        elif norm == 'adain':
            self.norm = AdaptiveInstanceNorm2d(norm_dim)
        elif norm == 'none':
            self.norm = None
        else:
            assert 0, 'Unsupported normalization: {}'.format(norm)
        if activation == 'elu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, 'Unsupported activation: {}'.format(activation)
        if snorm:
            self.conv = spectral_norm(nn.ConvTranspose2d(in_dim, out_dim,
                ks, st, bias=self.use_bias, padding=padding, output_padding
                =padding))
        else:
            self.conv = nn.ConvTranspose2d(in_dim, out_dim, ks, st, bias=
                self.use_bias, padding=padding, output_padding=padding)

    def forward(self, x, adain_params=None):
        if self.activation_first:
            if self.activation:
                x = self.activation(x)
            x = self.conv(x)
            if self.norm and not isinstance(self.norm, AdaptiveInstanceNorm2d):
                x = self.norm(x)
            elif isinstance(self.norm, AdaptiveInstanceNorm2d):
                x = self.norm(x, adain_params)
        else:
            x = self.conv(x)
            if self.norm and not isinstance(self.norm, AdaptiveInstanceNorm2d):
                x = self.norm(x)
            elif isinstance(self.norm, AdaptiveInstanceNorm2d):
                x = self.norm(x, adain_params)
            if self.activation:
                x = self.activation(x)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_dim': 4, 'out_dim': 4, 'ks': 4, 'st': 4}]
