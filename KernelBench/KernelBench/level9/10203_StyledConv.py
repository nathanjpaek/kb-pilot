from torch.autograd import Function
import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.functional import leaky_relu


def fused_leaky_relu(input_, bias, negative_slope=0.2, scale=2 ** 0.5):
    return scale * leaky_relu(input_ + bias[:input_.shape[1]],
        negative_slope, inplace=True)


def make_kernel(k):
    k = torch.tensor(k, dtype=torch.float32)
    if k.ndim == 1:
        k = k[None, :] * k[:, None]
    k /= k.sum()
    k = torch.flip(k, [0, 1])
    return k


def upfirdn2d_native(input, kernel, up_x, up_y, down_x, down_y, pad_x0,
    pad_x1, pad_y0, pad_y1):
    _, ch, _in_h, _in_w = input.shape
    kernel_h, kernel_w = kernel.shape
    assert up_y == up_x and up_y in [1, 2]
    if up_y == 2:
        w = input.new_zeros(2, 2)
        w[0, 0] = 1
        out = F.conv_transpose2d(input, w.view(1, 1, 2, 2).repeat(ch, 1, 1,
            1), groups=ch, stride=2)
    else:
        out = input
    out = F.pad(out, [pad_x0, pad_x1, pad_y0, pad_y1])
    out = F.conv2d(out, kernel.view(1, 1, kernel_h, kernel_w).repeat(ch, 1,
        1, 1), groups=ch)
    return out[:, :, ::down_y, ::down_x]


def upfirdn2d(input, kernel, up=1, down=1, pad=(0, 0)):
    if input.device.type == 'cpu':
        out = upfirdn2d_native(input, kernel, up, up, down, down, pad[0],
            pad[1], pad[0], pad[1])
    else:
        out = UpFirDn2d.apply(input, kernel, (up, up), (down, down), (pad[0
            ], pad[1], pad[0], pad[1]))
    return out


class FusedLeakyReLU(nn.Module):

    def __init__(self, channel, negative_slope=0.2, scale=2 ** 0.5):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(channel))
        self.negative_slope = negative_slope
        self.scale = scale

    def forward(self, x):
        return self.scale * leaky_relu(x + self.bias.reshape((1, -1, 1, 1))
            [:, :x.shape[1]], self.negative_slope, inplace=True)


class EqualLinear(nn.Module):

    def __init__(self, in_dim, out_dim, bias=True, bias_init=0, lr_mul=1.0,
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

    def forward(self, x):
        if self.activation:
            out = F.linear(x, self.weight * self.scale)
            if self.activation == 'lrelu':
                out = fused_leaky_relu(out, self.bias * self.lr_mul)
            else:
                raise NotImplementedError
        else:
            out = F.linear(x, self.weight * self.scale, bias=self.bias *
                self.lr_mul)
        return out

    def __repr__(self):
        return (
            f'{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]})'
            )


class UpFirDn2dBackward(Function):

    @staticmethod
    def forward(ctx, grad_output, kernel, grad_kernel, up, down, pad, g_pad,
        in_size, out_size):
        up_x, up_y = up
        down_x, down_y = down
        g_pad_x0, g_pad_x1, g_pad_y0, g_pad_y1 = g_pad
        grad_output = grad_output.reshape(-1, out_size[0], out_size[1], 1)
        grad_input = upfirdn2d_op.upfirdn2d(grad_output, grad_kernel,
            down_x, down_y, up_x, up_y, g_pad_x0, g_pad_x1, g_pad_y0, g_pad_y1)
        grad_input = grad_input.view(in_size[0], in_size[1], in_size[2],
            in_size[3])
        ctx.save_for_backward(kernel)
        pad_x0, pad_x1, pad_y0, pad_y1 = pad
        ctx.up_x = up_x
        ctx.up_y = up_y
        ctx.down_x = down_x
        ctx.down_y = down_y
        ctx.pad_x0 = pad_x0
        ctx.pad_x1 = pad_x1
        ctx.pad_y0 = pad_y0
        ctx.pad_y1 = pad_y1
        ctx.in_size = in_size
        ctx.out_size = out_size
        return grad_input

    @staticmethod
    def backward(ctx, gradgrad_input):
        kernel, = ctx.saved_tensors
        gradgrad_input = gradgrad_input.reshape(-1, ctx.in_size[2], ctx.
            in_size[3], 1)
        gradgrad_out = upfirdn2d_op.upfirdn2d(gradgrad_input, kernel, ctx.
            up_x, ctx.up_y, ctx.down_x, ctx.down_y, ctx.pad_x0, ctx.pad_x1,
            ctx.pad_y0, ctx.pad_y1)
        gradgrad_out = gradgrad_out.view(ctx.in_size[0], ctx.in_size[1],
            ctx.out_size[0], ctx.out_size[1])
        return gradgrad_out, None, None, None, None, None, None, None, None


class UpFirDn2d(Function):

    @staticmethod
    def forward(ctx, input, kernel, up, down, pad):
        up_x, up_y = up
        down_x, down_y = down
        pad_x0, pad_x1, pad_y0, pad_y1 = pad
        kernel_h, kernel_w = kernel.shape
        _batch, channel, in_h, in_w = input.shape
        ctx.in_size = input.shape
        input = input.reshape(-1, in_h, in_w, 1)
        ctx.save_for_backward(kernel, torch.flip(kernel, [0, 1]))
        out_h = (in_h * up_y + pad_y0 + pad_y1 - kernel_h) // down_y + 1
        out_w = (in_w * up_x + pad_x0 + pad_x1 - kernel_w) // down_x + 1
        ctx.out_size = out_h, out_w
        ctx.up = up_x, up_y
        ctx.down = down_x, down_y
        ctx.pad = pad_x0, pad_x1, pad_y0, pad_y1
        g_pad_x0 = kernel_w - pad_x0 - 1
        g_pad_y0 = kernel_h - pad_y0 - 1
        g_pad_x1 = in_w * up_x - out_w * down_x + pad_x0 - up_x + 1
        g_pad_y1 = in_h * up_y - out_h * down_y + pad_y0 - up_y + 1
        ctx.g_pad = g_pad_x0, g_pad_x1, g_pad_y0, g_pad_y1
        out = upfirdn2d_op.upfirdn2d(input, kernel, up_x, up_y, down_x,
            down_y, pad_x0, pad_x1, pad_y0, pad_y1)
        out = out.view(-1, channel, out_h, out_w)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        kernel, grad_kernel = ctx.saved_tensors
        grad_input = UpFirDn2dBackward.apply(grad_output, kernel,
            grad_kernel, ctx.up, ctx.down, ctx.pad, ctx.g_pad, ctx.in_size,
            ctx.out_size)
        return grad_input, None, None, None, None


class Blur(nn.Module):

    def __init__(self, kernel, pad, upsample_factor=1):
        super().__init__()
        kernel = make_kernel(kernel)
        if upsample_factor > 1:
            kernel = kernel * upsample_factor ** 2
        self.register_buffer('kernel', kernel)
        self.pad = pad

    def forward(self, x):
        out = upfirdn2d(x, self.kernel, pad=self.pad)
        return out


class ModulatedConv2d(nn.Module):

    def __init__(self, in_channel, out_channel, kernel_size, style_dim,
        demodulate=True, upsample=False, downsample=False, blur_kernel=(1, 
        3, 3, 1)):
        super().__init__()
        self.eps = 1e-08
        self.kernel_size = kernel_size
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.upsample = upsample
        self.downsample = downsample
        assert not downsample, 'Downsample is not implemented yet!'
        self.modulation = EqualLinear(style_dim, in_channel, bias_init=1)
        self.demodulate = demodulate
        if upsample:
            factor = 2
            p = len(blur_kernel) - factor - (kernel_size - 1)
            self.blur = Blur(blur_kernel, pad=((p + 1) // 2 + factor - 1, p //
                2 + 1), upsample_factor=factor)
        self.scale = 1 / math.sqrt(in_channel * kernel_size ** 2)
        self.padding = kernel_size // 2
        self.weight = nn.Parameter(torch.randn(1, out_channel, in_channel,
            kernel_size, kernel_size))

    def __repr__(self):
        return (
            f'{self.__class__.__name__}({self.in_channel}, {self.out_channel}, {self.kernel_size}, upsample={self.upsample}, downsample={self.downsample})'
            )

    def forward(self, x, style):
        batch, in_channel, height, width = x.shape
        style = self.modulation(style)
        style = style.view(batch, 1, -1, 1, 1)
        first_k_oup = self.first_k_oup if hasattr(self, 'first_k_oup'
            ) and self.first_k_oup is not None else self.weight.shape[1]
        assert first_k_oup <= self.weight.shape[1]
        weight = self.weight
        weight = weight[:, :first_k_oup, :in_channel].contiguous()
        weight = self.scale * weight * style[:, :, :in_channel]
        if self.demodulate:
            weight = weight * torch.rsqrt(weight.pow(2).sum([2, 3, 4],
                keepdim=True) + self.eps)
        if self.upsample:
            x = x.view(1, batch * in_channel, height, width)
            weight = weight.transpose(1, 2)
            weight = weight.reshape(weight.shape[0] * weight.shape[1],
                weight.shape[2], weight.shape[3], weight.shape[4])
            out = F.conv_transpose2d(x, weight, padding=0, stride=2, groups
                =batch)
            out = out.view(batch, -1, out.shape[-2], out.shape[-1])
            out = self.blur(out)
        else:
            x = x.contiguous().view(1, batch * in_channel, height, width)
            weight = weight.view(weight.shape[0] * weight.shape[1], weight.
                shape[2], weight.shape[3], weight.shape[4])
            out = F.conv2d(x, weight, padding=self.padding, groups=batch)
            out = out.view(batch, -1, out.shape[-2], out.shape[-1])
        return out


class NoiseInjection(nn.Module):

    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1))

    def forward(self, image, noise=None):
        if noise is None:
            batch, _, height, width = image.shape
            noise = image.new_empty(batch, 1, height, width).normal_()
        return image + self.weight * noise


class StyledConv(nn.Module):

    def __init__(self, in_channel, out_channel, kernel_size, style_dim,
        upsample=False, blur_kernel=(1, 3, 3, 1), demodulate=True,
        activation='lrelu'):
        super().__init__()
        self.conv = ModulatedConv2d(in_channel, out_channel, kernel_size,
            style_dim, upsample=upsample, blur_kernel=blur_kernel,
            demodulate=demodulate)
        self.noise = NoiseInjection()
        if activation == 'lrelu':
            self.activate = FusedLeakyReLU(out_channel)
        else:
            raise NotImplementedError

    def forward(self, x, style, noise=None):
        out = self.conv(x, style)
        out = self.noise(out, noise=noise)
        out = self.activate(out)
        return out


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channel': 4, 'out_channel': 4, 'kernel_size': 4,
        'style_dim': 4}]
