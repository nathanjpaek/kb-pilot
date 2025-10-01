import torch
import torch.nn as nn


class BReLU(nn.Module):
    """
    Biased ReLU
    
    BReLU(x) = ReLU(x) + b
    
    Shape:
        -Input: (N, *)
        -Output: (N, *), same shape as the input
    
    Parameters:
        -in_features: number of input features
        -b: fixed parameter (bias like for relu)
    
    Examples:
        >>> input = torch.randn(300, 6)
        >>> afunc = BReLU(input.shape[1], b = 1.0e-8)
        >>> x = afunc(input)
    """

    def __init__(self, in_features, b):
        super(BReLU, self).__init__()
        self.in_features = in_features
        self.b = nn.Parameter(torch.tensor(b), requires_grad=False)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(x) + self.b


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_features': 4, 'b': 4}]
