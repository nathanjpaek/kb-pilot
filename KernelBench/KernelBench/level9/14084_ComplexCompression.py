from torch.autograd import Function
import torch
from torch import Tensor
from typing import Tuple
from torch import nn
from torch.nn.parameter import Parameter


class angle_re_im(Function):
    """Similar to torch.angle but robustify the gradient for zero magnitude."""

    @staticmethod
    def forward(ctx, re: 'Tensor', im: 'Tensor'):
        ctx.save_for_backward(re, im)
        return torch.atan2(im, re)

    @staticmethod
    def backward(ctx, grad: 'Tensor') ->Tuple[Tensor, Tensor]:
        re, im = ctx.saved_tensors
        grad_inv = grad / (re.square() + im.square()).clamp_min_(1e-10)
        return -im * grad_inv, re * grad_inv


class ComplexCompression(nn.Module):

    def __init__(self, n_freqs: 'int', init_value: 'float'=0.3):
        super().__init__()
        self.register_parameter('c', Parameter(torch.full((n_freqs,),
            init_value), requires_grad=True))

    def forward(self, x: 'Tensor'):
        x_abs = (x[:, 0].square() + x[:, 1].square()).clamp_min(1e-10).pow(self
            .c)
        x_ang = angle_re_im.apply(x[:, 0], x[:, 1])
        x = torch.stack((x_abs * torch.cos(x_ang), x_abs * torch.sin(x_ang)
            ), dim=1)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'n_freqs': 4}]
