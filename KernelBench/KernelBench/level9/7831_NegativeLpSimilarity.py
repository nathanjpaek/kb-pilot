import torch
from torch import nn
from abc import abstractmethod
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


class NegativeLpSimilarity(Similarity):
    """Negative l_p distance similarity."""

    def __init__(self, p: 'float'=2.0):
        """
        Initialize the similarity.

        :param p:
            The parameter p for the l_p distance. See also: torch.cdist
        """
        super().__init__()
        self.p = p

    def forward(self, x: 'torch.Tensor', y: 'torch.Tensor') ->torch.Tensor:
        return -torch.cdist(x, y, p=self.p)


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
