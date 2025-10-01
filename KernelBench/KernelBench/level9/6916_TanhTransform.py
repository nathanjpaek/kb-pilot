import torch
import torch.nn as nn


def arctanh(x, eps=1e-06):
    """
    Calculates the inverse hyperbolic tangent.
    """
    x *= 1.0 - eps
    return torch.log((1 + x) / (1 - x)) * 0.5


class TanhTransform(nn.Module):
    """
    Computes the tanh transform used to
    remove box constraints from C&W paper

    NOTE: This reparamterization trick is
    highly numerically unstable even for small-ish
    values so should really only be used
    for inputs that are bounded above or below
    by relatively small values

    Args:
        xmin (float or torch.Tensor):
            the lower bound for input values
            should either be a float or broadcastable
            with the input tensor where each element
            in the tensor corresponds to the lower
            bound of an input feature

        xmax (float or torch.Tensor):
            the lower bound for input values
            should either be a float or broadcastable
            with the input tensor where each element
            in the tensor corresponds to the upper
            bound of an input feature

    """

    def __init__(self, xmin=0, xmax=1):
        super(TanhTransform, self).__init__()
        delta = xmax - xmin
        self.delta_2 = delta / 2
        self.xmax = xmax
        self.xmin = xmin

    def forward(self, x):
        out = (x.tanh() + 1) * self.delta_2 + self.xmin
        return out

    def invert_forward(self, x):
        z = (x - self.xmin) / self.delta_2 - 1
        return arctanh(z)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
