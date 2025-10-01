import torch
from torch import nn
import torch.nn.functional as F
import torch.utils.data.distributed


def _max_except_dim(input, dim):
    maxed = input
    for axis in range(input.ndimension() - 1, dim, -1):
        maxed, _ = maxed.max(axis, keepdim=True)
    for axis in range(dim - 1, -1, -1):
        maxed, _ = maxed.max(axis, keepdim=True)
    return maxed


def _norm_except_dim(w, norm_type, dim):
    if norm_type == 1 or norm_type == 2:
        return torch.norm_except_dim(w, norm_type, dim)
    elif norm_type == float('inf'):
        return _max_except_dim(w, dim)


def operator_norm_settings(domain, codomain):
    if domain == 1 and codomain == 1:
        max_across_input_dims = True
        norm_type = 1
    elif domain == 1 and codomain == 2:
        max_across_input_dims = True
        norm_type = 2
    elif domain == 1 and codomain == float('inf'):
        max_across_input_dims = True
        norm_type = float('inf')
    elif domain == 2 and codomain == float('inf'):
        max_across_input_dims = False
        norm_type = 2
    elif domain == float('inf') and codomain == float('inf'):
        max_across_input_dims = False
        norm_type = 1
    else:
        raise ValueError('Unknown combination of domain "{}" and codomain "{}"'
            .format(domain, codomain))
    return max_across_input_dims, norm_type


def _logit(p):
    p = torch.max(torch.ones(1) * 0.1, torch.min(torch.ones(1) * 0.9, p))
    return torch.log(p + 1e-10) + torch.log(1 - p + 1e-10)


class LipNormConv2d(nn.Conv2d):
    """Lipschitz constant defined using operator norms."""

    def __init__(self, in_channels, out_channels, kernel_size, stride,
        padding, bias=True, coeff=0.97, domain=float('inf'), codomain=float
        ('inf'), local_constraint=True, **unused_kwargs):
        del unused_kwargs
        super(LipNormConv2d, self).__init__(in_channels, out_channels,
            kernel_size, stride, padding, bias)
        self.coeff = coeff
        self.domain = domain
        self.codomain = codomain
        self.local_constraint = local_constraint
        max_across_input_dims, self.norm_type = operator_norm_settings(self
            .domain, self.codomain)
        self.max_across_dim = 1 if max_across_input_dims else 0
        with torch.no_grad():
            w_scale = _norm_except_dim(self.weight, self.norm_type, dim=
                self.max_across_dim)
            if not self.local_constraint:
                w_scale = w_scale.max()
            self.scale = nn.Parameter(_logit(w_scale / self.coeff))

    def compute_weight(self):
        w_scale = _norm_except_dim(self.weight, self.norm_type, dim=self.
            max_across_dim)
        if not self.local_constraint:
            w_scale = w_scale.max()
        return self.weight / w_scale * torch.sigmoid(self.scale)

    def forward(self, input):
        weight = self.compute_weight()
        return F.conv2d(input, weight, self.bias, self.stride, self.padding,
            1, 1)

    def extra_repr(self):
        s = super(LipNormConv2d, self).extra_repr()
        return s + ', coeff={}, domain={}, codomain={}, local={}'.format(self
            .coeff, self.domain, self.codomain, self.local_constraint)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4,
        'stride': 1, 'padding': 4}]
