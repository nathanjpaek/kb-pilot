import torch
from torch import nn
from abc import abstractmethod
import torch.utils.data


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


class MultiplicationComposition(Composition):
    """Element-wise multiplication, a.k.a. Hadamard product."""

    def forward(self, x: 'torch.Tensor', y: 'torch.Tensor') ->torch.Tensor:
        return x * y


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
