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


class Conv2dBlock(nn.Module):

    def __init__(self, in_dim, out_dim, ks, st, padding=0, norm='none',
        activation='elu', pad_type='zero', use_bias=True, activation_first=
        False, snorm=False):
        super().__init__()
        self.use_bias = use_bias
        self.activation_first = activation_first
        if pad_type == 'reflect':
            self.pad = nn.ReflectionPad2d(padding)
        elif pad_type == 'replicate':
            self.pad = nn.ReplicationPad2d(padding)
        elif pad_type == 'zero':
            self.pad = nn.ZeroPad2d(padding)
        else:
            assert 0, 'Unsupported padding type: {}'.format(pad_type)
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
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'elu':
            self.activation = nn.ELU()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, 'Unsupported activation: {}'.format(activation)
        if snorm:
            self.conv = spectral_norm(nn.Conv2d(in_dim, out_dim, ks, st,
                bias=self.use_bias))
        else:
            self.conv = nn.Conv2d(in_dim, out_dim, ks, st, bias=self.use_bias)

    def forward(self, x, adain_params=None):
        if self.activation_first:
            if self.activation:
                x = self.activation(x)
            x = self.conv(self.pad(x))
            if self.norm and not isinstance(self.norm, AdaptiveInstanceNorm2d):
                x = self.norm(x)
            elif isinstance(self.norm, AdaptiveInstanceNorm2d):
                x = self.norm(x, adain_params)
        else:
            x = self.conv(self.pad(x))
            if self.norm and not isinstance(self.norm, AdaptiveInstanceNorm2d):
                x = self.norm(x)
            elif isinstance(self.norm, AdaptiveInstanceNorm2d):
                x = self.norm(x, adain_params)
            if self.activation:
                x = self.activation(x)
        return x


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


class ResBlock(nn.Module):

    def __init__(self, dim_in, dim_out, norm='in', activation='elu',
        pad_type='zero', upsampling=False, stride=1, snorm=False):
        super(ResBlock, self).__init__()
        self.norm = norm
        self.model = nn.ModuleList()
        if upsampling:
            self.conv1 = Conv2dTransposeBlock(dim_in, dim_out, 3, 2, 1,
                norm=self.norm, activation=activation, snorm=snorm)
            self.conv2 = Conv2dBlock(dim_out, dim_out, 3, 1, 1, norm=self.
                norm, activation='none', pad_type=pad_type, snorm=snorm)
        else:
            self.conv1 = Conv2dBlock(dim_in, dim_out, 3, stride, 1, norm=
                self.norm, activation=activation, pad_type=pad_type, snorm=
                snorm)
            self.conv2 = Conv2dBlock(dim_out, dim_out, 3, 1, 1, norm=self.
                norm, activation='none', pad_type=pad_type, snorm=snorm)
        self.convolve_res = dim_in != dim_out or upsampling or stride != 1
        if self.convolve_res:
            if not upsampling:
                self.res_conv = Conv2dBlock(dim_in, dim_out, 3, stride, 1,
                    norm='in', activation=activation, pad_type=pad_type,
                    snorm=snorm)
            else:
                self.res_conv = Conv2dTransposeBlock(dim_in, dim_out, 3, 2,
                    1, norm='in', activation=activation, snorm=snorm)

    def forward(self, x, adain_params=None):
        residual = x
        if self.convolve_res:
            residual = self.res_conv(residual)
        out = self.conv1(x, adain_params)
        out = self.conv2(out, adain_params)
        out += residual
        return out


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'dim_in': 4, 'dim_out': 4}]
