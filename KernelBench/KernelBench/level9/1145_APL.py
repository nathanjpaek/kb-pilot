import torch
from torch import nn
from torch.nn.parameter import Parameter


class APL(nn.Module):
    """
    Implementation of APL (ADAPTIVE PIECEWISE LINEAR UNITS) unit:

        .. math::

            APL(x_i) = max(0,x) + \\sum_{s=1}^{S}{a_i^s * max(0, -x + b_i^s)}

    with trainable parameters a and b, parameter S should be set in advance.

    Shape:
        - Input: (N, *) where * means, any number of additional
          dimensions
        - Output: (N, *), same shape as the input

    Parameters:
        - S: hyperparameter, number of hinges to be set in advance
        - a: trainable parameter, control the slopes of the linear segments
        - b: trainable parameter, determine the locations of the hinges

    References:
        - See APL paper:
        https://arxiv.org/pdf/1412.6830.pdf

    Examples:
        >>> a1 = apl(256, S = 1)
        >>> x = torch.randn(256)
        >>> x = a1(x)
    """

    def __init__(self, in_features, S, a=None, b=None):
        """
        Initialization.
        INPUT:
            - in_features: shape of the input
            - S (int): number of hinges
            - a - value for initialization of parameter, which controls the slopes of the linear segments
            - b - value for initialization of parameter, which determines the locations of the hinges
            a, b are initialized randomly by default
        """
        super(APL, self).__init__()
        self.in_features = in_features
        self.S = S
        if a is None:
            self.a = Parameter(torch.randn((S, in_features), dtype=torch.
                float, requires_grad=True))
        else:
            self.a = a
        if b is None:
            self.b = Parameter(torch.randn((S, in_features), dtype=torch.
                float, requires_grad=True))
        else:
            self.b = b

    def forward(self, x):
        """
        Forward pass of the function
        """
        output = x.clamp(min=0)
        for s in range(self.S):
            t = -x + self.b[s]
            output += self.a[s] * t.clamp(min=0)
        return output


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_features': 4, 'S': 4}]
