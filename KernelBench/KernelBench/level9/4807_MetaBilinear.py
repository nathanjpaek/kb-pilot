import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict


class MetaModule(nn.Module):
    """
    Base class for PyTorch meta-learning modules. These modules accept an
    additional argument `params` in their `forward` method.

    Notes
    -----
    Objects inherited from `MetaModule` are fully compatible with PyTorch
    modules from `torch.nn.Module`. The argument `params` is a dictionary of
    tensors, with full support of the computation graph (for differentiation).
    """

    def meta_named_parameters(self, prefix='', recurse=True):
        gen = self._named_members(lambda module: module._parameters.items() if
            isinstance(module, MetaModule) else [], prefix=prefix, recurse=
            recurse)
        for elem in gen:
            yield elem

    def meta_parameters(self, recurse=True):
        for name, param in self.meta_named_parameters(recurse=recurse):
            yield param


class MetaBilinear(nn.Bilinear, MetaModule):
    __doc__ = nn.Bilinear.__doc__

    def forward(self, input1, input2, params=None):
        if params is None:
            params = OrderedDict(self.named_parameters())
        bias = params.get('bias', None)
        return F.bilinear(input1, input2, params['weight'], bias)


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in1_features': 4, 'in2_features': 4, 'out_features': 4}]
