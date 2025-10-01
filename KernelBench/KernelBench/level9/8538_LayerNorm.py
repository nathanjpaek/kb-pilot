import torch
from typing import Callable
from typing import Tuple
import torch.utils.data
from typing import Union
import torch.nn
import torch.cuda
import torch.backends.cudnn


def batch_elementwise(input: 'torch.Tensor', param: 'torch.Tensor', op:
    'Callable[[torch.Tensor, torch.Tensor], torch.Tensor]', input_batch_dim:
    'int'=0, pndim: 'int'=1) ->torch.Tensor:
    """
    Do elementwise operation in groups.

    :param input: input, any shape, [..., Ci, Cj, ...]
    :param param: the parameter, shape [N, Ci, Cj....], in which case B % N == 0, or [Ci, Cj....]
    :param input_batch_dim: which dimension is the batch in the input
    :param op: the operation to perform
    :param pndim: number of parameter dimensions without batch
    :return: input with the op performed, the same shape as input
    """
    if param.ndim == pndim + 1:
        param = param.squeeze(0)
    if param.ndim == pndim:
        return op(input, param)
    assert param.ndim == pndim + 1
    assert input.shape[input_batch_dim] % param.shape[0] == 0
    input_r = input.view(*input.shape[:input_batch_dim], param.shape[0], -1,
        *input.shape[input_batch_dim + 1:])
    param_r = param.view(*([1] * input_batch_dim), param.shape[0], *([1] *
        (input_r.ndim - input_batch_dim - param.ndim)), *param.shape[1:])
    return op(input_r, param_r).view_as(input)


def batch_bias_add(*args, **kwargs) ->torch.Tensor:
    """
    Batch add bias to the inputs.

    For more details, see batch_elementwise
    """
    return batch_elementwise(*args, op=lambda a, b: a + b, **kwargs)


def batch_const_mul(*args, **kwargs) ->torch.Tensor:
    """
    Batch multiplies bias to the inputs.

    For more details, see batch_elementwise
    """
    return batch_elementwise(*args, op=lambda a, b: a * b, **kwargs)


class MaskedModule(torch.nn.Module):
    pass


class LayerNorm(MaskedModule):

    def __init__(self, normalized_shape: 'Union[int, Tuple[int]]', eps=1e-05):
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = normalized_shape,
        self.gamma = torch.nn.Parameter(torch.ones(*normalized_shape))
        self.beta = torch.nn.Parameter(torch.zeros(*normalized_shape))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return batch_bias_add(batch_const_mul((x - mean) / (std + self.eps),
            self.gamma), self.beta)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'normalized_shape': 4}]
