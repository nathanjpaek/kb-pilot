import math
import torch
import torch.utils.data
from itertools import product as product
from math import sqrt as sqrt
import torch.nn


class NormalizedLinear(torch.nn.Module):
    """
    A advanced Linear layer which supports weight normalization or cosine normalization.

    """

    def __init__(self, in_features, out_features, bias=False, feat_norm=
        True, scale_mode='learn', scale_init=1.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.feat_norm = feat_norm
        self.scale_mode = scale_mode
        self.scale_init = scale_init
        self.weight = torch.nn.Parameter(torch.Tensor(out_features,
            in_features))
        if bias:
            self.bias = torch.nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        if self.scale_mode == 'constant':
            self.scale = scale_init
        elif self.scale_mode == 'learn':
            self.scale = torch.nn.Parameter(torch.ones(1) * scale_init)
        else:
            raise NotImplementedError

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight
                )
            bound = 1 / math.sqrt(fan_in)
            torch.nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, inputs):
        """
        Args:
            inputs (torch.Tensor): (N, C)
        Return:
            output (torch.Tensor): (N, D)
        """
        if self.feat_norm:
            inputs = torch.nn.functional.normalize(inputs, dim=1)
        output = inputs.mm(torch.nn.functional.normalize(self.weight, dim=1
            ).t())
        output = self.scale * output
        return output

    def extra_repr(self):
        s = 'in_features={in_features}, out_features={out_features}'
        if self.bias is None:
            s += ', bias=False'
        s += ', feat_norm={feat_norm}'
        s += ', scale_mode={scale_mode}'
        s += ', scale_init={scale_init}'
        return s.format(**self.__dict__)


def get_inputs():
    return [torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'in_features': 4, 'out_features': 4}]
