import torch
import numpy as np
from typing import Tuple
import torch.nn.functional as F
from typing import cast


def scale(old_value, old_min, old_max, new_min, new_max):
    old_range = old_max - old_min
    new_range = new_max - new_min
    new_value = (old_value - old_min) * new_range / old_range + new_min
    return new_value


class LinearFBSP(torch.nn.Module):

    def __init__(self, out_features: 'int', bias: 'bool'=True, normalized:
        'bool'=False):
        super(LinearFBSP, self).__init__()
        self.out_features = out_features
        self.normalized = normalized
        self.eps = 1e-08
        default_dtype = torch.get_default_dtype()
        self.register_parameter('m', torch.nn.Parameter(torch.zeros(self.
            out_features, dtype=default_dtype)))
        self.register_parameter('fb', torch.nn.Parameter(torch.ones(self.
            out_features, dtype=default_dtype)))
        self.register_parameter('fc', torch.nn.Parameter(torch.arange(self.
            out_features, dtype=default_dtype)))
        self.register_parameter('bias', torch.nn.Parameter(torch.normal(0.0,
            0.5, (self.out_features, 2), dtype=default_dtype) if bias else
            cast(torch.nn.Parameter, None)))
        self.m.register_hook(lambda grad: grad / (torch.norm(grad, p=float(
            'inf')) + self.eps))
        self.fb.register_hook(lambda grad: grad / (torch.norm(grad, p=float
            ('inf')) + self.eps))
        self.fc.register_hook(lambda grad: grad / (torch.norm(grad, p=float
            ('inf')) + self.eps))

    @staticmethod
    def power(x1: 'torch.Tensor', x2: 'torch.Tensor') ->torch.Tensor:
        magnitudes = (x1[..., 0] ** 2 + x1[..., 1] ** 2) ** 0.5
        phases = x1[..., 1].atan2(x1[..., 0])
        power_real = x2[..., 0]
        power_imag = x2[..., 1]
        mag_out = (magnitudes ** 2) ** (0.5 * power_real) * torch.exp(-
            power_imag * phases)
        return mag_out.unsqueeze(-1) * torch.stack(((power_real * phases + 
            0.5 * power_imag * (magnitudes ** 2).log()).cos(), (power_real *
            phases + 0.5 * power_imag * (magnitudes ** 2).log()).sin()), dim=-1
            )

    @staticmethod
    def sinc(x: 'torch.Tensor') ->torch.Tensor:
        return torch.where(cast(torch.Tensor, x == 0), torch.ones_like(x), 
            torch.sin(x) / x)

    def _materialize_weights(self, x: 'torch.Tensor') ->Tuple[torch.Tensor,
        bool]:
        x_is_complex = x.shape[-1] == 2
        in_features = x.shape[-1 - int(x_is_complex)]
        t = np.pi * torch.linspace(-1.0, 1.0, in_features, dtype=x.dtype,
            device=x.device).reshape(1, -1, 1) + self.eps
        m = self.m.reshape(-1, 1, 1)
        fb = self.fb.reshape(-1, 1, 1)
        fc = self.fc.reshape(-1, 1, 1)
        kernel = torch.cat((torch.cos(fc * t), -torch.sin(fc * t)), dim=-1)
        scale = fb.sqrt()
        win = self.sinc(fb * t / (m + self.eps))
        win = self.power(torch.cat((win, torch.zeros_like(win)), dim=-1),
            torch.cat((m, torch.zeros_like(m)), dim=-1))
        weights = scale * torch.cat((win[..., :1] * kernel[..., :1] - win[
            ..., 1:] * kernel[..., 1:], win[..., :1] * kernel[..., 1:] + 
            win[..., 1:] * kernel[..., :1]), dim=-1)
        if self.normalized:
            weights = weights / in_features ** 0.5
        return weights, x_is_complex

    def forward(self, x: 'torch.Tensor') ->Tuple[torch.Tensor, torch.Tensor]:
        weights, x_is_complex = self._materialize_weights(x)
        if x_is_complex:
            x = torch.stack((F.linear(x[..., 0], weights[..., 0]) - F.
                linear(x[..., 1], weights[..., 1]), F.linear(x[..., 0],
                weights[..., 1]) + F.linear(x[..., 1], weights[..., 0])),
                dim=-1)
        else:
            x = torch.stack((F.linear(x, weights[..., 0]), F.linear(x,
                weights[..., 1])), dim=-1)
        if self.bias is not None and self.bias.numel(
            ) == self.out_features * 2:
            x = x + self.bias
        return x, weights

    def extra_repr(self) ->str:
        return 'out_features={}, bias={}, normalized={}'.format(self.
            out_features, self.bias is not None and self.bias.numel() == 
            self.out_features * 2, self.normalized)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'out_features': 4}]
