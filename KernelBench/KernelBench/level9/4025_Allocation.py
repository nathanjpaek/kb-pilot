from torch.nn import Module
import torch
from torch.nn import functional as F
from torch.nn import Linear


class Allocation(Module):
    """Determines allocation probability for each of the bidders given an input.
    
    Args:
        in_features: size of each input sample
        bidders: number of bidders, which governs the size of each output sample
    Shape:
        - Input: :math:`(N, *, H_{in})` where :math:`*` means any number of
          additional dimensions and :math:`H_{in} = \\text{in\\_features}`
        - Output: :math:`(N, *, H_{out})` where all but the last dimension
          are the same shape as the input and :math:`H_{out} = \\text{bidders}`.

    Examples::
        >>> m = Allocation(20, 30)
        >>> input = torch.randn(128, 20)
        >>> allocation = m(input)
        >>> print(allocation.size())
        torch.Size([128, 30])
    """
    __constants__ = ['in_features', 'bidders']

    def __init__(self, in_features, bidders):
        super(Allocation, self).__init__()
        self.in_features = in_features
        self.bidders = bidders
        self.linear = Linear(in_features, bidders + 1)

    def forward(self, x):
        return F.softmax(self.linear(x), dim=1)[:, 0:self.bidders]


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_features': 4, 'bidders': 4}]
