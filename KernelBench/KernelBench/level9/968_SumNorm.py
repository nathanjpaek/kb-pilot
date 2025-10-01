import torch
import torch.nn as nn


class SumNorm(nn.Module):
    """
    Normalize dividing by the sum
    
    Shape:
        -Input: (N, *)
        -Output: (N, *), same shape as the input
    
    Parameters:
        -in_features: number of input features
        -dim(int): A dimension along witch sum will be computed
    
    Examples:
        >>> input = torch.randn(300, 4)
        >>> afunc = SumNorm(input.shape[1], dim = 1)
        >>> x = afunc(input)
        
    """

    def __init__(self, in_features, dim=1):
        super(SumNorm, self).__init__()
        self.in_features = in_features
        self.dim = dim

    def forward(self, x):
        return x / x.sum(dim=self.dim).view(x.shape[0], 1)


def get_inputs():
    return [torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'in_features': 4}]
