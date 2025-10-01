from torch.nn import Module
import math
import torch
from torch.nn.parameter import Parameter
import torch.nn.functional as F


class NAC(Module):
    """Neural Accumulator: :math:`y = Wx` where :math:`W = \\tanh(\\hat{W}) * \\sigma(\\hat{M})`

    Args:
        in_features: size of each input sample
        out_features: size of each output sample

    Shape:
        - Input: :math:`(N, *, in\\_features)` where :math:`*` means any number of
          additional dimensions
        - Output: :math:`(N, *, out\\_features)` where all but the last dimension
          are the same shape as the input.

    Attributes:
        weight_tanh: the learnable weights of the module of shape
            `(out_features x in_features)`
        weight_sigma: the learnable weights of the module of shape
            `(out_features x in_features)`

    Examples:
        >>> m = NAC(20, 30)
        >>> input = torch.randn(128, 20)
        >>> output = m(input)
        >>> print(output.size())
        torch.Size([128, 30])

        >>> m = NAC(2, 1)
        >>> _ = m.weight_tanh.data.fill_(4), m.weight_sigma.data.fill_(4)
        >>> m.weight
        tensor([[0.9814, 0.9814]], grad_fn=<ThMulBackward>)
        >>> input = torch.Tensor([[0, 1], [2, 5], [-1, 4]])
        >>> output = m(input)
        >>> output
        tensor([[0.9814],
                [6.8695],
                [2.9441]], grad_fn=<MmBackward>)
    """

    def __init__(self, in_features, out_features):
        super(NAC, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight_tanh = Parameter(torch.Tensor(out_features, in_features))
        self.weight_sigma = Parameter(torch.Tensor(out_features, in_features))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.out_features)
        self.weight_tanh.data.uniform_(-stdv, stdv)
        self.weight_sigma.data.uniform_(-stdv, stdv)

    @property
    def weight(self):
        """
        Effective weight of NAC
        :return:
        """
        return torch.tanh(self.weight_tanh) * torch.sigmoid(self.weight_sigma)

    def forward(self, input):
        return F.linear(input, weight=self.weight)

    def extra_repr(self):
        return 'in_features={}, out_features={}'.format(self.in_features,
            self.out_features)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_features': 4, 'out_features': 4}]
