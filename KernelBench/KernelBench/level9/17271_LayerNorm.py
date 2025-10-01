import torch
from torch import Tensor
from torch.nn import Parameter
from torch.nn import LayerNorm
from typing import Optional
import torch.fx
from typing import Any
import torch.utils.data
from inspect import Parameter
from torch.nn.parameter import Parameter


def maybe_num_nodes(edge_index, num_nodes=None):
    if num_nodes is not None:
        return num_nodes
    elif isinstance(edge_index, Tensor):
        return int(edge_index.max()) + 1 if edge_index.numel() > 0 else 0
    else:
        return max(edge_index.size(0), edge_index.size(1))


def constant(value: 'Any', fill_value: 'float'):
    if isinstance(value, Tensor):
        value.data.fill_(fill_value)
    else:
        for v in (value.parameters() if hasattr(value, 'parameters') else []):
            constant(v, fill_value)
        for v in (value.buffers() if hasattr(value, 'buffers') else []):
            constant(v, fill_value)


def zeros(value: 'Any'):
    constant(value, 0.0)


def ones(tensor: 'Any'):
    constant(tensor, 1.0)


def degree(index, num_nodes: 'Optional[int]'=None, dtype: 'Optional[int]'=None
    ):
    """Computes the (unweighted) degree of a given one-dimensional index
    tensor.

    Args:
        index (LongTensor): Index tensor.
        num_nodes (int, optional): The number of nodes, *i.e.*
            :obj:`max_val + 1` of :attr:`index`. (default: :obj:`None`)
        dtype (:obj:`torch.dtype`, optional): The desired data type of the
            returned tensor.

    :rtype: :class:`Tensor`
    """
    N = maybe_num_nodes(index, num_nodes)
    out = torch.zeros((N,), dtype=dtype, device=index.device)
    one = torch.ones((index.size(0),), dtype=out.dtype, device=out.device)
    return out.scatter_add_(0, index, one)


class LayerNorm(torch.nn.Module):
    """Applies layer normalization over each individual example in a batch
    of node features as described in the `"Layer Normalization"
    <https://arxiv.org/abs/1607.06450>`_ paper

    .. math::
        \\mathbf{x}^{\\prime}_i = \\frac{\\mathbf{x} -
        \\textrm{E}[\\mathbf{x}]}{\\sqrt{\\textrm{Var}[\\mathbf{x}] + \\epsilon}}
        \\odot \\gamma + \\beta

    The mean and standard-deviation are calculated across all nodes and all
    node channels separately for each object in a mini-batch.

    Args:
        in_channels (int): Size of each input sample.
        eps (float, optional): A value added to the denominator for numerical
            stability. (default: :obj:`1e-5`)
        affine (bool, optional): If set to :obj:`True`, this module has
            learnable affine parameters :math:`\\gamma` and :math:`\\beta`.
            (default: :obj:`True`)
    """

    def __init__(self, in_channels, eps=1e-05, affine=True):
        super(LayerNorm, self).__init__()
        self.in_channels = in_channels
        self.eps = eps
        if affine:
            self.weight = Parameter(torch.Tensor([in_channels]))
            self.bias = Parameter(torch.Tensor([in_channels]))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        ones(self.weight)
        zeros(self.bias)

    def forward(self, x: 'Tensor', batch: 'OptTensor'=None) ->Tensor:
        """"""
        if batch is None:
            x = x - x.mean()
            out = x / (x.std(unbiased=False) + self.eps)
        else:
            batch_size = int(batch.max()) + 1
            norm = degree(batch, batch_size, dtype=x.dtype).clamp_(min=1)
            norm = norm.mul_(x.size(-1)).view(-1, 1)
            mean = scatter(x, batch, dim=0, dim_size=batch_size, reduce='add'
                ).sum(dim=-1, keepdim=True) / norm
            x = x - mean[batch]
            var = scatter(x * x, batch, dim=0, dim_size=batch_size, reduce=
                'add').sum(dim=-1, keepdim=True)
            var = var / norm
            out = x / (var + self.eps).sqrt()[batch]
        if self.weight is not None and self.bias is not None:
            out = out * self.weight + self.bias
        return out

    def __repr__(self):
        return f'{self.__class__.__name__}({self.in_channels})'


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 4}]
