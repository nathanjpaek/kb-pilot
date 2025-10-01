import torch
import torch.nn as nn
import torch.nn.functional as F


class PositiveLinear(nn.Linear):
    """Applies a transformation to the incoming data of the following form: :math:`y_i = xlog(exp(A)+1)^T`
        where log and exp are elementwise operations.

        Args:
            in_features: size of each input sample
            out_features: size of each output sample
            bias: If set to False, the layer will not learn an additive bias.
                Default: ``True``

        Shape:
            - Input: :math:`(N, *, in\\_features)` where :math:`*` means any number of
              additional dimensions
            - Output: :math:`(N, *, out\\_features)` where all but the last dimension
              are the same shape as the input.

        Attributes:
            weight: the learnable weights of the module of shape
                `(out_features x in_features)`
            bias:   the learnable bias of the module of shape `(out_features)`

        Examples::

            >>> m = nn.PositiveLinear(20, 30)
            >>> input = torch.randn(128, 20)
            >>> output = m(input)
            >>> print(output.size())
        """

    def forward(self, input):
        transformed_weight = torch.clamp(self.weight, min=0)
        torch.clamp(self.bias, min=0)
        return F.linear(input, transformed_weight, self.bias)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_features': 4, 'out_features': 4}]
