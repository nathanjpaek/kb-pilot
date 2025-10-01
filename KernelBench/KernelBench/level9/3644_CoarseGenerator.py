import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm as spectral_norm_fn
from torch.nn.utils import weight_norm as weight_norm_fn


def gen_conv(input_dim, output_dim, kernel_size=3, stride=1, padding=0,
    rate=1, activation='elu', gated=False):
    """ 
    Convolutions used in the generator.
    
    Args:
        input_dim (int): number of input channels.
        output_dim (int): number of output features.
        kernel_size (int): kernel size of convolutional filters.
        stride (int): convolutional stride.
        padding (int): padding for convolution.
        rate (int): dilation rate of dilated convolution.
        activation (string): activation on computed features.
        gated (bool): boolean deciding on making convolutions "gated".
        
    Return:
        (tensor): result from convolution
    """
    if gated:
        conv2 = Conv2dBlockGated(input_dim, output_dim, kernel_size, stride,
            conv_padding=padding, dilation=rate, activation=activation)
    else:
        conv2 = Conv2dBlock(input_dim, output_dim, kernel_size, stride,
            conv_padding=padding, dilation=rate, activation=activation)
    return conv2


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


class Conv2dBlockGated(nn.Module):

    def __init__(self, input_dim, output_dim, kernel_size, stride, padding=
        0, conv_padding=0, dilation=1, weight_norm='none', norm='none',
        activation='relu', pad_type='zero', transpose=False):
        super(Conv2dBlockGated, self).__init__()
        self.use_bias = True
        self._output_dim = output_dim
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
            self.gate = nn.ConvTranspose2d(input_dim, output_dim,
                kernel_size, stride, padding=conv_padding, output_padding=
                conv_padding, dilation=dilation, bias=self.use_bias)
        else:
            self.conv = nn.Conv2d(input_dim, output_dim, kernel_size,
                stride, padding=conv_padding, dilation=dilation, bias=self.
                use_bias)
            self.gate = nn.Conv2d(input_dim, output_dim, kernel_size,
                stride, padding=conv_padding, dilation=dilation, bias=self.
                use_bias)
        if self.weight_norm:
            self.conv = self.weight_norm(self.conv)
            self.gate = self.weight_norm(self.gate)

    def forward(self, x):
        if self.pad:
            feat = self.conv(self.pad(x))
            gate = self.gate(self.pad(x))
        else:
            feat = self.conv(x)
            gate = self.gate(x)
        if self.norm:
            feat = self.norm(feat)
            gate = self.norm(gate)
        if self.activation is None or self._output_dim == 3:
            return feat
        feat = self.activation(feat)
        gate = torch.sigmoid(gate)
        return feat * gate


class CoarseGenerator(nn.Module):

    def __init__(self, input_dim, cnum, gated=False, use_cuda=True, device=0):
        super(CoarseGenerator, self).__init__()
        self.use_cuda = use_cuda
        self.device = device
        self.conv1 = gen_conv(input_dim + 2, cnum, 5, 1, 2, gated=gated)
        self.conv2_downsample = gen_conv(cnum, cnum * 2, 3, 2, 1, gated=gated)
        self.conv3 = gen_conv(cnum * 2, cnum * 2, 3, 1, 1, gated=gated)
        self.conv4_downsample = gen_conv(cnum * 2, cnum * 4, 3, 2, 1, gated
            =gated)
        self.conv5 = gen_conv(cnum * 4, cnum * 4, 3, 1, 1, gated=gated)
        self.conv6 = gen_conv(cnum * 4, cnum * 4, 3, 1, 1, gated=gated)
        self.conv7_atrous = gen_conv(cnum * 4, cnum * 4, 3, 1, 2, rate=2,
            gated=gated)
        self.conv8_atrous = gen_conv(cnum * 4, cnum * 4, 3, 1, 4, rate=4,
            gated=gated)
        self.conv9_atrous = gen_conv(cnum * 4, cnum * 4, 3, 1, 8, rate=8,
            gated=gated)
        self.conv10_atrous = gen_conv(cnum * 4, cnum * 4, 3, 1, 16, rate=16,
            gated=gated)
        self.conv11 = gen_conv(cnum * 4, cnum * 4, 3, 1, 1, gated=gated)
        self.conv12 = gen_conv(cnum * 4, cnum * 4, 3, 1, 1, gated=gated)
        self.conv13 = gen_conv(cnum * 4, cnum * 2, 3, 1, 1, gated=gated)
        self.conv14 = gen_conv(cnum * 2, cnum * 2, 3, 1, 1, gated=gated)
        self.conv15 = gen_conv(cnum * 2, cnum, 3, 1, 1, gated=gated)
        self.conv16 = gen_conv(cnum, cnum // 2, 3, 1, 1, gated=gated)
        self.conv17 = gen_conv(cnum // 2, input_dim, 3, 1, 1, activation=
            'none', gated=gated)

    def forward(self, x, mask):
        ones = torch.ones(x.size(0), 1, x.size(2), x.size(3), device=x.device)
        x = torch.cat([x, ones, mask], dim=1)
        x = self.conv1(x)
        x = self.conv2_downsample(x)
        x = self.conv3(x)
        x = self.conv4_downsample(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7_atrous(x)
        x = self.conv8_atrous(x)
        x = self.conv9_atrous(x)
        x = self.conv10_atrous(x)
        x = self.conv11(x)
        x = self.conv12(x)
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = self.conv13(x)
        x = self.conv14(x)
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = self.conv15(x)
        x = self.conv16(x)
        x = self.conv17(x)
        x_stage1 = torch.clamp(x, -1.0, 1.0)
        return x_stage1


def get_inputs():
    return [torch.rand([4, 1, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'input_dim': 4, 'cnum': 4}]
