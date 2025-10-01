import torch
from torch import nn
import torch.utils.data
import torch.nn.functional
import torch.autograd


class FeedForward(nn.Module):
    """
    ### Position-wise Feed Forward Layer $	ext{F\\small{FW}}$

    This consists of two linear layers and an activation in the middle.
    """

    def __init__(self, d_model: 'int', d_ff: 'int'):
        """
        * `d_model` is the number of features in transformer embeddings
        * `d_ff` is the number features in the hidden layer
        """
        super().__init__()
        self.lin1 = nn.Linear(d_model, d_ff)
        self.lin2 = nn.Linear(d_ff, d_model)
        self.act = nn.ReLU()
        self.norm = nn.LayerNorm(d_model)

    def forward(self, h: 'torch.Tensor'):
        """
        `h` are the embeddings of shape `[batch_size, seq_len, d_model]`
        """
        h_res = h
        h = self.norm(h)
        h = self.lin1(h)
        h = self.act(h)
        h = self.lin2(h)
        return h + h_res


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'d_model': 4, 'd_ff': 4}]
