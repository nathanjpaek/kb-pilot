from torch.autograd import Function
import torch
import torch.utils.data.distributed
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data


def quantize(input, nbit):
    return Quantizer.apply(input, nbit)


def dorefa_a(input, nbit_a):
    return quantize(torch.clamp(0.1 * input, 0, 1), nbit_a)


def scale_sign(input):
    return ScaleSigner.apply(input)


def dorefa_w(w, nbit_w):
    if nbit_w == 1:
        w = scale_sign(w)
    else:
        w = torch.tanh(w)
        w = w / (2 * torch.max(torch.abs(w))) + 0.5
        w = 2 * quantize(w, nbit_w) - 1
    return w


class Quantizer(Function):

    @staticmethod
    def forward(ctx, input, nbit):
        scale = 2 ** nbit - 1
        return torch.round(input * scale) / scale

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


class ScaleSigner(Function):
    """take a real value x, output sign(x)*E(|x|)"""

    @staticmethod
    def forward(ctx, input):
        return torch.sign(input) * torch.mean(torch.abs(input))

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


class QuanConv(nn.Conv2d):
    """docstring for QuanConv"""

    def __init__(self, in_channels, out_channels, kernel_size, quan_name_w=
        'dorefa', quan_name_a='dorefa', nbit_w=1, nbit_a=1, stride=1,
        padding=0, dilation=1, groups=1, bias=True):
        super(QuanConv, self).__init__(in_channels, out_channels,
            kernel_size, stride, padding, dilation, groups, bias)
        self.nbit_w = nbit_w
        self.nbit_a = nbit_a
        name_w_dict = {'dorefa': dorefa_w}
        name_a_dict = {'dorefa': dorefa_a}
        self.quan_w = name_w_dict[quan_name_w]
        self.quan_a = name_a_dict[quan_name_a]

    def forward(self, input):
        if self.nbit_w < 32:
            w = self.quan_w(self.weight, self.nbit_w)
        else:
            w = self.weight
        if self.nbit_a < 32:
            x = self.quan_a(input, self.nbit_a)
        else:
            x = input
        output = F.conv2d(x, w, self.bias, self.stride, self.padding, self.
            dilation, self.groups)
        return output


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}]
