import torch
import torch.nn as nn


def swish_func(x, beta=1.0, inplace=False):
    """
    "Swish: a Self-Gated Activation Function"
    Searching for Activation Functions (https://arxiv.org/abs/1710.05941)

    If beta=1 applies the Sigmoid Linear Unit (SiLU) function element-wise
    If beta=0, Swish becomes the scaled linear function (identity
      activation) f(x) = x/2
    As beta -> ∞, the sigmoid component converges to approach a 0-1 function
      (unit step), and multiplying that by x gives us f(x)=2max(0,x), which
      is the ReLU multiplied by a constant factor of 2, so Swish becomes like
      the ReLU function.

    Including beta, Swish can be loosely viewed as a smooth function that
      nonlinearly interpolate between identity (linear) and ReLU function.
      The degree of interpolation can be controlled by the model if beta is
      set as a trainable parameter.

    Alt: 1.78718727865 * (x * sigmoid(x) - 0.20662096414)
    """
    if inplace:
        result = x.clone()
        torch.sigmoid_(beta * x)
        x *= result
        return x
    return x * torch.sigmoid(beta * x)


class Swish(nn.Module):
    __constants__ = ['beta', 'slope', 'inplace']

    def __init__(self, beta=1.0, slope=1.67653251702, inplace=False):
        """
        Shape:
        - Input: (N, *) where * means, any number of additional
          dimensions
        - Output: (N, *), same shape as the input
        """
        super(Swish, self).__init__()
        self.inplace = inplace
        self.beta = torch.nn.Parameter(torch.tensor(beta))
        self.beta.requiresGrad = True
        self.slope = slope / 2

    def forward(self, x):
        """
        # Disabled, using inplace causes:
        # "RuntimeError: one of the variables needed for gradient computation has been modified by an inplace operation"
        if self.inplace:
            input.mul_(torch.sigmoid(self.beta*input))
            return 2 * self.slope * input
        else:
            return 2 * self.slope * swish_func(input, self.beta)
        """
        return 2 * self.slope * swish_func(x, self.beta, self.inplace)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
