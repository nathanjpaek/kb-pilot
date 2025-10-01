import torch
import torch.nn as nn


class ComplexConv(nn.Module):

    def __init__(self, rank, in_channels, out_channels, kernel_size, stride
        =1, padding=0, output_padding=0, dilation=1, groups=1, bias=True,
        normalize_weight=False, epsilon=1e-07, conv_transposed=False):
        super(ComplexConv, self).__init__()
        self.rank = rank
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        self.conv_transposed = conv_transposed
        self.conv_args = {'in_channels': self.in_channels, 'out_channels':
            self.out_channels, 'kernel_size': self.kernel_size, 'stride':
            self.stride, 'padding': self.padding, 'groups': self.groups,
            'bias': self.bias}
        if self.conv_transposed:
            self.conv_args['output_padding'] = self.output_padding
        else:
            self.conv_args['dilation'] = self.dilation
        self.conv_func = {(1): nn.Conv1d if not self.conv_transposed else
            nn.ConvTranspose1d, (2): nn.Conv2d if not self.conv_transposed else
            nn.ConvTranspose2d, (3): nn.Conv3d if not self.conv_transposed else
            nn.ConvTranspose3d}[self.rank]
        self.real_conv = self.conv_func(**self.conv_args)
        self.imag_conv = self.conv_func(**self.conv_args)

    def forward(self, input):
        """
            assume complex input z = x + iy needs to be convolved by complex filter h = a + ib
            where Output O = z * h, where * is convolution operator, then O = x*a + i(x*b)+ i(y*a) - y*b
            so we need to calculate each of the 4 convolution operations in the previous equation,
            one simple way to implement this as two conolution layers, one layer for the real weights (a)
            and the other for imaginary weights (b), this can be done by concatenating both real and imaginary
            parts of the input and convolve over both of them as follows:
            c_r = [x; y] * a , and  c_i= [x; y] * b, so that
            O_real = c_r[real] - c_i[real], and O_imag = c_r[imag] - c_i[imag]
        """
        ndims = input.ndimension()
        input_real = input.narrow(ndims - 1, 0, 1).squeeze(ndims - 1)
        input_imag = input.narrow(ndims - 1, 1, 1).squeeze(ndims - 1)
        output_real = self.real_conv(input_real) - self.imag_conv(input_imag)
        output_imag = self.real_conv(input_imag) + self.imag_conv(input_real)
        output = torch.stack([output_real, output_imag], dim=ndims - 1)
        return output


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'rank': 2, 'in_channels': 4, 'out_channels': 4,
        'kernel_size': 4}]
