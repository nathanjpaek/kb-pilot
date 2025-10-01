import torch
from torch import nn
import torch.nn.modules.loss
from scipy.sparse import *


class MeanEmbedding(nn.Module):
    """Mean embedding class.
    """

    def __init__(self):
        super(MeanEmbedding, self).__init__()

    def forward(self, emb, len_):
        """Compute average embeddings.

        Parameters
        ----------
        emb : torch.Tensor
            The input embedding tensor.
        len_ : torch.Tensor
            The sequence length tensor.

        Returns
        -------
        torch.Tensor
            The average embedding tensor.
        """
        return torch.sum(emb, dim=-2) / len_


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
