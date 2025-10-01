import torch
import torch.nn as nn
import torch.utils.data
import torch.multiprocessing
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
        summed = torch.sum(emb, dim=-2)
        len_ = len_.unsqueeze(-1).expand_as(summed).float()
        return summed / len_


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4])]


def get_init_inputs():
    return [[], {}]
