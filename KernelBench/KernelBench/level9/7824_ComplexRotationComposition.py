import torch
from torch import nn
from abc import abstractmethod
import torch.utils.data


def _to_complex(x: 'torch.Tensor') ->torch.Tensor:
    """View real tensor as complex."""
    return torch.view_as_complex(x.view(*x.shape[:-1], -1, 2))


def _to_real(x: 'torch.Tensor') ->torch.Tensor:
    """View complex tensor as real."""
    x = torch.view_as_real(x)
    return x.view(*x.shape[:-2], -1)


def _complex_multiplication(x: 'torch.Tensor', y: 'torch.Tensor', y_norm:
    'bool'=False) ->torch.Tensor:
    """Element-wise multiplication as complex numbers."""
    x = _to_complex(x)
    y = _to_complex(y)
    if y_norm:
        y = y / y.abs().clamp_min(1e-08)
    x = x * y
    return _to_real(x)


class Composition(nn.Module):
    """A base class for compositions."""

    @abstractmethod
    def forward(self, x: 'torch.Tensor', y: 'torch.Tensor') ->torch.Tensor:
        """
        Compose two batches vectors.


        .. note ::
            The two batches have to be of broadcastable shape.

        :param x: shape: s_x
            The first batch of vectors.
        :param y: shape: s_y
            The second batch of vectors.

        :return: shape: s
            The compositionm, where `s` is the broadcasted shape.
        """
        raise NotImplementedError


class ComplexRotationComposition(Composition):
    """Composition by rotation in complex plane."""

    def forward(self, x: 'torch.Tensor', y: 'torch.Tensor') ->torch.Tensor:
        return _complex_multiplication(x, y, y_norm=True)


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
