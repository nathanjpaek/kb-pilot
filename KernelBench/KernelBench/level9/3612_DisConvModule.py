import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm as spectral_norm_fn
from torch.nn.utils import weight_norm as weight_norm_fn


def dis_conv(input_dim, output_dim, kernel_size=5, stride=2, padding=0,
    rate=1, activation='lrelu', weight_norm='none'):
    return Conv2dBlock(input_dim, output_dim, kernel_size, stride,
        conv_padding=padding, dilation=rate, activation=activation,
        weight_norm=weight_norm)


class Conv2dBlock(nn.Module):

    def __init__(self, input_dim, output_dim, kernel_size, stride, padding=
        0, conv_padding=0, dilation=1, weight_norm='none', norm='none',
        activation='relu', pad_type='zero', transpose=False):
        super(Conv2dBlock, self).__init__()
        self.use_bias = True
        if pad_type == 'reflect':
            self.pad = nn.ReflectionPad2d(padding)
        elif pad_type == 'replicate':
            self.pad = nn.ReplicationPad2d(padding)
        elif pad_type == 'zero':
            self.pad = nn.ZeroPad2d(padding)
        elif pad_type == 'none':
            self.pad = None
        else:
            assert 0, 'Unsupported padding type: {}'.format(pad_type)
        norm_dim = output_dim
        if norm == 'bn':
            self.norm = nn.BatchNorm2d(norm_dim)
        elif norm == 'in':
            self.norm = nn.InstanceNorm2d(norm_dim)
        elif norm == 'none':
            self.norm = None
        else:
            assert 0, 'Unsupported normalization: {}'.format(norm)
        if weight_norm == 'sn':
            self.weight_norm = spectral_norm_fn
        elif weight_norm == 'wn':
            self.weight_norm = weight_norm_fn
        elif weight_norm == 'none':
            self.weight_norm = None
        else:
            assert 0, 'Unsupported normalization: {}'.format(weight_norm)
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'elu':
            self.activation = nn.ELU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU(inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, 'Unsupported activation: {}'.format(activation)
        if transpose:
            self.conv = nn.ConvTranspose2d(input_dim, output_dim,
                kernel_size, stride, padding=conv_padding, output_padding=
                conv_padding, dilation=dilation, bias=self.use_bias)
        else:
            self.conv = nn.Conv2d(input_dim, output_dim, kernel_size,
                stride, padding=conv_padding, dilation=dilation, bias=self.
                use_bias)
        if self.weight_norm:
            self.conv = self.weight_norm(self.conv)

    def forward(self, x):
        if self.pad:
            x = self.conv(self.pad(x))
        else:
            x = self.conv(x)
        if self.norm:
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        return x


class DisConvModule(nn.Module):

    def __init__(self, input_dim, cnum, weight_norm='none', use_cuda=True,
        device_id=0):
        super(DisConvModule, self).__init__()
        self.use_cuda = use_cuda
        self.device_id = device_id
        self.conv1 = dis_conv(input_dim, cnum, 5, 2, 2, weight_norm=weight_norm
            )
        self.conv2 = dis_conv(cnum, cnum * 2, 5, 2, 2, weight_norm=weight_norm)
        self.conv3 = dis_conv(cnum * 2, cnum * 4, 5, 2, 2, weight_norm=
            weight_norm)
        self.conv4 = dis_conv(cnum * 4, cnum * 4, 5, 2, 2, weight_norm=
            weight_norm)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'input_dim': 4, 'cnum': 4}]
