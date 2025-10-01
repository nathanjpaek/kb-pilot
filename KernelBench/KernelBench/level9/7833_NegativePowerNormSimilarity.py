import torch
from torch import nn
from abc import abstractmethod
from typing import Union
import torch.utils.data


class Similarity(nn.Module):
    """Base class for similarity functions."""

    @abstractmethod
    def forward(self, x: 'torch.Tensor', y: 'torch.Tensor') ->torch.Tensor:
        """
        Compute pair-wise similarities.

        :param x: shape: (*, n, d)
            The first vectors.
        :param y: shape: (*, m, d)
            The second vectors.

        :return: shape: (*, n, m)
            The similarity values.
        """
        raise NotImplementedError

    def one_to_one(self, x: 'torch.Tensor', y: 'torch.Tensor') ->torch.Tensor:
        """
        Compute batched one-to-one similarities.

        :param x: shape: (*, d)
            The first vectors.
        :param y: shape: (*, d)
            The second vectors.

        :return: shape: (*)
            The similarity values.
        """
        return self(x.unsqueeze(dim=-2), y.unsqueeze(dim=-2)).squeeze(dim=-1
            ).squeeze(dim=-1)


class NegativePowerNormSimilarity(Similarity):
    """Negative power norm: -\\|x - y\\|_p^p."""

    def __init__(self, p: 'Union[int, float]'=2):
        """
        Initialize the similarity.

        :param p:
            The parameter p for the p-norm.
        """
        super().__init__()
        self.p = p

    def forward(self, x: 'torch.Tensor', y: 'torch.Tensor') ->torch.Tensor:
        return -(x.unsqueeze(dim=-2) - y.unsqueeze(dim=-3)).pow(self.p).sum(dim
            =-1)


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
