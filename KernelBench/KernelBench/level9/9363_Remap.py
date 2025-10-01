import torch
import numpy as np
import torch.nn as nn
from abc import abstractmethod
from typing import Union
from typing import Tuple
from typing import List


class BaseModel(nn.Module):
    """
    Base class for all models
    """

    @abstractmethod
    def forward(self, *inputs):
        """
        Forward pass logic

        :return: Model output
        """
        raise NotImplementedError

    def __str__(self):
        """
        Model prints with number of trainable parameters
        """
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + '\nTrainable parameters: {}'.format(params)


class Remap(BaseModel):
    """
    Basic layer for element-wise remapping of values from one range to another.
    """
    in_range: 'Tuple[float, float]'
    out_range: 'Tuple[float, float]'

    def __init__(self, in_range: 'Union[Tuple[float, float], List[float]]',
        out_range: 'Union[Tuple[float, float], List[float]]'):
        assert len(in_range) == len(out_range) and len(in_range) == 2
        super(BaseModel, self).__init__()
        self.in_range = tuple(in_range)
        self.out_range = tuple(out_range)

    def forward(self, x: 'torch.Tensor') ->torch.Tensor:
        return torch.div(torch.mul(torch.add(x, -self.in_range[0]), self.
            out_range[1] - self.out_range[0]), self.in_range[1] - self.
            in_range[0] + self.out_range[0])


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_range': [4, 4], 'out_range': [4, 4]}]
