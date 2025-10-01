from torch.autograd import Function
import math
import torch
from torch import nn
import torch.nn.functional as F


def fused_leaky_relu(input, bias, negative_slope=0.2, scale=2 ** 0.5):
    return FusedLeakyReLUFunction.apply(input, bias, negative_slope, scale)


class FusedLeakyReLUFunctionBackward(Function):

    @staticmethod
    def forward(ctx, grad_output, out, negative_slope, scale):
        ctx.save_for_backward(out)
        ctx.negative_slope = negative_slope
        ctx.scale = scale
        empty = grad_output.new_empty(0)
        grad_input = fused.fused_bias_act(grad_output, empty, out, 3, 1,
            negative_slope, scale)
        dim = [0]
        if grad_input.ndim > 2:
            dim += list(range(2, grad_input.ndim))
        grad_bias = grad_input.sum(dim).detach()
        return grad_input, grad_bias

    @staticmethod
    def backward(ctx, gradgrad_input, gradgrad_bias):
        out, = ctx.saved_tensors
        gradgrad_out = fused.fused_bias_act(gradgrad_input, gradgrad_bias,
            out, 3, 1, ctx.negative_slope, ctx.scale)
        return gradgrad_out, None, None, None


class FusedLeakyReLUFunction(Function):

    @staticmethod
    def forward(ctx, input, bias, negative_slope, scale):
        if input.dtype != bias.dtype:
            input = input.type(bias.dtype)
        empty = input.new_empty(0)
        out = fused.fused_bias_act(input, bias, empty, 3, 0, negative_slope,
            scale)
        ctx.save_for_backward(out)
        ctx.negative_slope = negative_slope
        ctx.scale = scale
        return out

    @staticmethod
    def backward(ctx, grad_output):
        out, = ctx.saved_tensors
        grad_input, grad_bias = FusedLeakyReLUFunctionBackward.apply(
            grad_output, out, ctx.negative_slope, ctx.scale)
        return grad_input, grad_bias, None, None


class EqualLinear(nn.Module):

    def __init__(self, in_dim, out_dim, bias=True, bias_init=0, lr_mul=1,
        activation=None):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_dim, in_dim).div_(lr_mul))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_dim).fill_(bias_init))
        else:
            self.bias = None
        self.activation = activation
        self.scale = 1 / math.sqrt(in_dim) * lr_mul
        self.lr_mul = lr_mul

    def forward(self, input):
        if self.activation:
            out = F.linear(input, self.weight * self.scale)
            out = fused_leaky_relu(out, self.bias * self.lr_mul)
        else:
            out = F.linear(input, self.weight * self.scale, bias=self.bias *
                self.lr_mul)
        return out

    def __repr__(self):
        return (
            f'{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]})'
            )


class NoiseInjection(nn.Module):

    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1))

    def forward(self, image, noise=None):
        if noise is None:
            batch, _, height, width = image.shape
            noise = image.new_empty(batch, 1, height, width).normal_()
        return image + self.weight * noise


class ModulatedSiren2d(nn.Module):

    def __init__(self, in_channel, out_channel, style_dim, demodulate=True,
        is_first=False, omega_0=30):
        super().__init__()
        self.eps = 1e-08
        self.kernel_size = 1
        self.in_channel = in_channel
        self.out_channel = out_channel
        fan_in = in_channel
        self.scale = 1 / math.sqrt(fan_in)
        if is_first:
            self.demod_scale = 3 * fan_in
        else:
            self.demod_scale = omega_0 ** 2 / 2
        self.omega0 = omega_0
        self.weight = nn.Parameter(torch.randn(1, out_channel, in_channel, 
            1, 1))
        self.modulation = EqualLinear(style_dim, in_channel, bias_init=1)
        self.demodulate = demodulate

    def __repr__(self):
        return (
            f'{self.__class__.__name__}({self.in_channel}, {self.out_channel})'
            )

    def forward(self, input, style):
        batch, in_channel, height, width = input.shape
        style = self.modulation(style)
        style = style.view(batch, 1, in_channel, 1, 1)
        weight = self.scale * self.weight * style
        if self.demodulate:
            demod = torch.rsqrt(weight.pow(2).sum([2, 3, 4]) * self.
                demod_scale + 1e-08)
            weight = weight * demod.view(batch, self.out_channel, 1, 1, 1)
        weight = weight.view(batch * self.out_channel, in_channel, 1, 1)
        input = input.view(1, batch * in_channel, height, width)
        out = F.conv2d(input, weight, groups=batch)
        _, _, height, width = out.shape
        out = out.view(batch, self.out_channel, height, width)
        out = out * self.omega0
        return out


class StyledSiren(nn.Module):

    def __init__(self, in_channel, out_channel, style_dim, demodulate=True,
        is_first=False, omega_0=30.0):
        super().__init__()
        self.conv = ModulatedSiren2d(in_channel, out_channel, style_dim,
            demodulate=demodulate, is_first=is_first, omega_0=omega_0)
        self.noise = NoiseInjection()

    def forward(self, input, style, noise=None):
        out = self.conv(input, style)
        out = self.noise(out, noise=noise)
        out = torch.sin(out)
        return out


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'in_channel': 4, 'out_channel': 4, 'style_dim': 4}]
