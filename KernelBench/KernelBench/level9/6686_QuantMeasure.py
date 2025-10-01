import torch
from torch import nn
from torch.autograd.function import InplaceFunction


def quantize(x, num_bits=8, min_value=None, max_value=None, num_chunks=None,
    stochastic=False, inplace=False, quantize=False, layer_num=-1, multi=
    False, index=[], is_act=False):
    return UniformQuantize().apply(x, num_bits, min_value, max_value,
        stochastic, inplace, num_chunks, False, quantize, layer_num, multi,
        index, is_act)


class UniformQuantize(InplaceFunction):

    @classmethod
    def forward(cls, ctx, input, num_bits=8, min_value=None, max_value=None,
        stochastic=False, inplace=False, num_chunks=None, out_half=False,
        quantize=False, layer_num=-1, multi=False, index=[], is_act=False):
        if is_act:
            multi = False
        num_chunks = num_chunks = input.shape[0
            ] if num_chunks is None else num_chunks
        if min_value is None or max_value is None:
            B = input.shape[0]
            y = input.view(B // num_chunks, -1)
        if min_value is None:
            min_value = y.min(-1)[0].mean(-1)
        if max_value is None:
            max_value = y.max(-1)[0].mean(-1)
        ctx.inplace = inplace
        ctx.num_bits = num_bits
        ctx.min_value = min_value
        ctx.max_value = max_value
        ctx.stochastic = stochastic
        if ctx.inplace:
            ctx.mark_dirty(input)
            output = input
        else:
            output = input.clone()
        if multi:
            bit_max = 8
            for i in range(bit_max):
                if len(index[layer_num][i]) == 0:
                    continue
                else:
                    idx = index[layer_num][i]
                min_value = output[idx].min()
                max_value = output[idx].max()
                qmin = 0.0
                qmax = 2.0 ** (1 + i) - 1.0
                scale = (max_value - min_value) / (qmax - qmin)
                scale = max(scale, 1e-08)
                output[idx] = output[idx].add_(-min_value).div_(scale).add_(
                    qmin)
                output[idx] = output[idx].clamp_(qmin, qmax).round_()
                output[idx] = output[idx].add_(-qmin).mul_(scale).add_(
                    min_value)
        else:
            min_value = output.min()
            max_value = output.max()
            qmin = 0.0
            qmax = 2.0 ** num_bits - 1.0
            scale = (max_value - min_value) / (qmax - qmin)
            scale = max(scale, 1e-08)
            output = output.add_(-min_value).div_(scale).add_(qmin)
            output = output.clamp_(qmin, qmax).round_()
            output = output.add_(-qmin).mul_(scale).add_(min_value)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output
        return (grad_input, None, None, None, None, None, None, None, None,
            None, None, None, None)


class QuantMeasure(nn.Module):
    """docstring for QuantMeasure."""

    def __init__(self, num_bits=8, quantize=False, momentum=0.1):
        super(QuantMeasure, self).__init__()
        self.register_buffer('running_min', torch.zeros(1))
        self.register_buffer('running_max', torch.zeros(1))
        self.momentum = momentum
        self.num_bits = num_bits
        self.quantize = quantize

    def forward(self, input):
        if self.training:
            min_value = input.detach().view(input.size(0), -1).min(-1)[0].mean(
                )
            max_value = input.detach().view(input.size(0), -1).max(-1)[0].mean(
                )
            self.running_min.mul_(self.momentum).add_(min_value * (1 - self
                .momentum))
            self.running_max.mul_(self.momentum).add_(max_value * (1 - self
                .momentum))
        else:
            min_value = self.running_min
            max_value = self.running_max
        return quantize(input, self.num_bits, min_value=float(min_value),
            max_value=float(max_value), num_chunks=16, quantize=self.
            quantize, is_act=True)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
