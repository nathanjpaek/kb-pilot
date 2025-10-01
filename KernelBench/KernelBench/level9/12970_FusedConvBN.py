import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.quantization
import torch.onnx
import torchaudio.functional as F
import torch.nn.parallel
import torch.utils.data
from torch.functional import F
import torch.fx
import torch.nn
import torch.optim
import torch.profiler


def unsqueeze_all(t):
    return t[None, :, None, None]


def batch_norm_backward(grad_out, X, sum, sqrt_var, N, eps):
    tmp = ((X - unsqueeze_all(sum) / N) * grad_out).sum(dim=(0, 2, 3))
    tmp *= -1
    d_denom = tmp / (sqrt_var + eps) ** 2
    d_var = d_denom / (2 * sqrt_var)
    d_mean_dx = grad_out / unsqueeze_all(sqrt_var + eps)
    d_mean_dx = unsqueeze_all(-d_mean_dx.sum(dim=(0, 2, 3)) / N)
    grad_input = X * unsqueeze_all(d_var * N)
    grad_input += unsqueeze_all(-d_var * sum)
    grad_input *= 2 / ((N - 1) * N)
    grad_input += d_mean_dx
    grad_input *= unsqueeze_all(sqrt_var + eps)
    grad_input += grad_out
    grad_input /= unsqueeze_all(sqrt_var + eps)
    return grad_input


def convolution_backward(grad_out, X, weight):
    grad_input = F.conv2d(X.transpose(0, 1), grad_out.transpose(0, 1)
        ).transpose(0, 1)
    grad_X = F.conv_transpose2d(grad_out, weight)
    return grad_X, grad_input


class FusedConvBN2DFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, X, conv_weight, eps=0.001):
        assert X.ndim == 4
        ctx.save_for_backward(X, conv_weight)
        X = F.conv2d(X, conv_weight)
        sum = X.sum(dim=(0, 2, 3))
        var = X.var(unbiased=True, dim=(0, 2, 3))
        N = X.numel() / X.size(1)
        sqrt_var = torch.sqrt(var)
        ctx.eps = eps
        ctx.sum = sum
        ctx.N = N
        ctx.sqrt_var = sqrt_var
        mean = sum / N
        denom = sqrt_var + eps
        out = X - unsqueeze_all(mean)
        out /= unsqueeze_all(denom)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        X, conv_weight = ctx.saved_tensors
        X_conv_out = F.conv2d(X, conv_weight)
        grad_out = batch_norm_backward(grad_out, X_conv_out, ctx.sum, ctx.
            sqrt_var, ctx.N, ctx.eps)
        grad_X, grad_input = convolution_backward(grad_out, X, conv_weight)
        return grad_X, grad_input, None, None, None, None, None


class FusedConvBN(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size,
        exp_avg_factor=0.1, eps=0.001, device=None, dtype=None):
        super(FusedConvBN, self).__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        weight_shape = out_channels, in_channels, kernel_size, kernel_size
        self.conv_weight = nn.Parameter(torch.empty(*weight_shape, **
            factory_kwargs))
        num_features = out_channels
        self.num_features = num_features
        self.eps = eps
        self.reset_parameters()

    def forward(self, X):
        return FusedConvBN2DFunction.apply(X, self.conv_weight, self.eps)

    def reset_parameters(self) ->None:
        nn.init.kaiming_uniform_(self.conv_weight, a=math.sqrt(5))


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}]
