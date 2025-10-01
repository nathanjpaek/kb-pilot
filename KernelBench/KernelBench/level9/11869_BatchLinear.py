import torch
from torch import nn
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


class BatchLinear(nn.Linear, MetaModule):
    """A linear meta-layer that can deal with batched weight matrices and biases, as for instance output by a
    hypernetwork."""
    __doc__ = nn.Linear.__doc__

    def forward(self, input, params=None):
        if params is None:
            params = OrderedDict(self.named_parameters())
        bias = params.get('bias', None)
        weight = params['weight']
        output = input.matmul(weight.permute(*[i for i in range(len(weight.
            shape) - 2)], -1, -2))
        output += bias.unsqueeze(-2)
        return output


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_features': 4, 'out_features': 4}]
