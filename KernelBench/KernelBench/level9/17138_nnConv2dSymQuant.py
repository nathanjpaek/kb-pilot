import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair


class SymmetricQuantizeDequantize(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, precision, clamp_val):
        ctx.save_for_backward(input)
        """
        Compute quantization step size. Mapping (-max_val, max_val) linearly to (-127,127)
        """
        use_max = True
        if use_max:
            max_val = torch.max(torch.abs(input))
        else:
            max_val = clamp_val
        delta = max_val / (2 ** (precision - 1) - 1)
        input_clamped = torch.clamp(input, -max_val, max_val)
        input_q = torch.round(input_clamped / delta)
        if precision == 8:
            input_q = input_q
        elif precision == 16:
            input_q = input_q
        else:
            input_q = input_q
        """
        Dequantize introducing a quantization error in the data
        """
        input_dq = input_q * delta
        input_dq = input_dq
        return input.copy_(input_dq)

    @staticmethod
    def backward(ctx, grad_output):
        _input, = ctx.saved_tensors
        return grad_output, None


class nnConv2dSymQuant(nn.Conv2d):
    """ 
    Computes 2d conv output
    Weights are quantized and dequantized introducing a quantization error
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
        padding=1, dilation=1, groups=1, bias=True, padding_mode='zeros',
        precision=-1, clamp_val=0.5):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super().__init__(in_channels, out_channels, kernel_size, stride,
            padding, dilation, groups, bias, padding_mode)
        self.precision = precision
        self.clamp_val = clamp_val

    def conv2d_forward(self, input, weight):
        return F.conv2d(input, weight, self.bias, self.stride, self.padding,
            self.dilation, self.groups)

    def forward(self, input):
        if self.precision > 0:
            quantWeight = SymmetricQuantizeDequantize.apply
            quantWeight(self.weight, self.precision, self.clamp_val)
        return self.conv2d_forward(input, self.weight)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}]
