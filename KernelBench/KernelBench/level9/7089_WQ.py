import torch
import torch.nn as nn


def stats_quant(x, nbit, qmode='symm', dequantize=True):
    z_typical = {'4bit': [0.077, 1.013], '8bit': [0.027, 1.114]}
    z = z_typical[f'{int(nbit)}bit']
    m = x.abs().mean()
    std = x.std()
    if qmode == 'symm':
        n_lv = 2 ** (nbit - 1) - 1
        alpha_w = 1 / z[0] * std - z[1] / z[0] * m
    elif qmode == 'asymm':
        n_lv = (2 ** nbit - 1) / 2
        alpha_w = 2 * m
    else:
        raise NotImplementedError
    x = x.clamp(-alpha_w.item(), alpha_w.item())
    scale = n_lv / alpha_w
    xq = x.mul(scale).round()
    if len(xq.unique()) > 2 ** nbit:
        xq = xq.clamp(-2 ** nbit // 2, 2 ** nbit // 2 - 1)
    if dequantize:
        xq = xq.div(scale)
    return xq, scale


class RoundQ(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, wbit, qmode):
        input_q, _scale = stats_quant(input, wbit, qmode)
        ctx.save_for_backward(input)
        return input_q

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input, None, None


class WQ(nn.Module):
    """
    Weight quantizer
    """

    def __init__(self, wbit, qmode='symm'):
        super(WQ, self).__init__()
        self.wbit = wbit
        self.qmode = qmode

    def forward(self, x):
        weight_q = RoundQ.apply(x, self.wbit, self.qmode)
        return weight_q

    def extra_repr(self):
        return super(WQ, self).extra_repr() + 'qmode={}'.format(self.qmode)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'wbit': 4}]
