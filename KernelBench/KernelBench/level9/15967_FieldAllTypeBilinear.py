import math
import torch
import torch.nn as nn
import torch.utils.data
from typing import Dict
from typing import Tuple
from abc import ABC
from abc import abstractmethod


class BaseLayer(nn.Module, ABC):
    """
    Base Layer for the torecsys module
    """

    def __init__(self, **kwargs):
        """
        Initializer for BaseLayer

        Args:
            **kwargs: kwargs
        """
        super(BaseLayer, self).__init__()

    @property
    @abstractmethod
    def inputs_size(self) ->Dict[str, Tuple[str, ...]]:
        """
        Get inputs size of the layer

        Returns:
            Dict[str, Tuple[str, ...]]: dictionary of inputs_size
        """
        raise NotImplementedError('not implemented')

    @property
    @abstractmethod
    def outputs_size(self) ->Dict[str, Tuple[str, ...]]:
        """
        Get outputs size of the layer

        Returns:
            Dict[str, Tuple[str, ...]]: dictionary of outputs_size
        """
        raise NotImplementedError('not implemented')


class FieldAllTypeBilinear(BaseLayer):
    """
    Applies a bilinear transformation to the incoming data: :math:`y = x_1 \\cdot W \\odot x_2 + b`
    
    Args:
        in1_features: size of each first input sample
        in2_features: size of each second input sample
        bias: If set to False, the layer will not learn an additive bias. Default: ``True``

    Shape:
        - Input1: :math:`(N, *, H_{in1})` where :math:`H_{in1}=\\text{in1\\_features}` and
            :math:`*` means any number of additional dimensions. All but the last dimension
            of the inputs should be the same
        - Input2: :math:`(N, *, H_{in2})` where :math:`H_{in2}=\\text{in2\\_features}`.
        - Output: :math:`(N, *, H_{out})` where :math:`H_{out}=\\text{out\\_features}`
            and all but the last dimension are the same shape as the input
    
    Examples::

        >>> m = FieldAllTypeBilinear(20, 20)
        >>> input1 = torch.randn(128, 10, 20)
        >>> input2 = torch.randn(128, 10, 3)
        >>> output = m(input1, input2)
        >>> print(output.size())
            torch.Size([128, 10, 3])
    """

    @property
    def inputs_size(self) ->Dict[str, Tuple[str, ...]]:
        return {'inputs1': ('B', 'NC2', 'E'), 'inputs2': ('B', 'NC2', 'E')}

    @property
    def outputs_size(self) ->Dict[str, Tuple[str, ...]]:
        return {'outputs': ('B', 'NC2', 'E')}
    __constants__ = ['in1_features', 'in2_features', 'bias']

    def __init__(self, in1_features, in2_features, bias=True):
        super(FieldAllTypeBilinear, self).__init__()
        self.in1_features = in1_features
        self.in2_features = in2_features
        self.weight = nn.Parameter(torch.Tensor(in1_features, in2_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(in2_features))
        else:
            self.register_parameter('bias', nn.Parameter(torch.tensor([0])))
        self.reset_parameters()

    def reset_parameters(self):
        bound = 1 / math.sqrt(self.weight.shape[0])
        nn.init.uniform_(self.weight, -bound, bound)
        if self.bias is not None:
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input1, input2):
        output = torch.mul(torch.matmul(input1, self.weight), input2)
        if self.bias is not None:
            output += self.bias
        return output

    def extra_repr(self):
        return (
            f'in1_features={self.in1_features}, in2_features={self.in2_features}, bias={self.bias is not None}'
            )


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in1_features': 4, 'in2_features': 4}]
