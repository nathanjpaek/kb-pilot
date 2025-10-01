import torch
from torch.nn import functional as F
import torch.nn as nn
import torch.utils.data
import torch.multiprocessing
import torch.nn.modules.loss
from scipy.sparse import *


def dropout(x, drop_prob, shared_axes=[], training=False):
    """
    Apply dropout to input tensor.
    Parameters
    ----------
    input_tensor: ``torch.FloatTensor``
        A tensor of shape ``(batch_size, ..., num_timesteps, embedding_dim)``
    Returns
    -------
    output: ``torch.FloatTensor``
        A tensor of shape ``(batch_size, ..., num_timesteps, embedding_dim)`` with dropout applied.
    """
    if drop_prob == 0 or drop_prob is None or not training:
        return x
    sz = list(x.size())
    for i in shared_axes:
        sz[i] = 1
    mask = x.new(*sz).bernoulli_(1.0 - drop_prob).div_(1.0 - drop_prob)
    mask = mask.expand_as(x)
    return x * mask


class InnerProductDecoder(nn.Module):
    """Decoder for using inner product for prediction."""

    def __init__(self, dropout, act=torch.sigmoid):
        super(InnerProductDecoder, self).__init__()
        self.dropout = dropout
        self.act = act

    def forward(self, z):
        z = F.dropout(z, self.dropout, training=self.training)
        adj = self.act(torch.mm(z, z.t()))
        return adj


def get_inputs():
    return [torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'dropout': 0.5}]
