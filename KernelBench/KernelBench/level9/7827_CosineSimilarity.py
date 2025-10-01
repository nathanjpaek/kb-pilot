import torch
from torch import nn
from abc import abstractmethod
import torch.utils.data
from torch.nn import functional


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


class DotProductSimilarity(Similarity):
    """Dot product similarity."""

    def forward(self, x: 'torch.Tensor', y: 'torch.Tensor') ->torch.Tensor:
        return x @ y.transpose(-2, -1)


class CosineSimilarity(DotProductSimilarity):
    """Cosine similarity."""

    def forward(self, x: 'torch.Tensor', y: 'torch.Tensor') ->torch.Tensor:
        x = functional.normalize(x, p=2, dim=-1)
        y = functional.normalize(y, p=2, dim=-1)
        return super().forward(x=x, y=y)


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
