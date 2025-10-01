import torch
from torch import nn
from typing import Optional


def old_style_broadcast(first: 'torch.Tensor', second: 'torch.Tensor', axis:
    'int') ->torch.Tensor:
    rank = len(first.shape)
    axis = axis + rank if axis < 0 else axis
    second_shape = [1] * axis + list(second.shape)
    second_shape = second_shape + [1] * (rank - len(second_shape))
    return second.view(second_shape)


class OnnxToTorchModule:
    """
    Marker class for onnx2torch modules.
    """
    pass


class OnnxPow(nn.Module, OnnxToTorchModule):

    def __init__(self, broadcast: 'Optional[int]'=None, axis:
        'Optional[int]'=None):
        super().__init__()
        self.axis = axis
        self.broadcast = broadcast

    def forward(self, input_tensor: 'torch.Tensor', exponent: 'torch.Tensor'
        ) ->torch.Tensor:
        if self.broadcast == 1 and self.axis is not None:
            exponent = old_style_broadcast(input_tensor, exponent, self.axis)
        return torch.pow(input_tensor, exponent)


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
