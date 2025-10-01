import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionwiseFeedForward(nn.Module):
    """A two-feed-forward-layer module.

    Parameters
    ----------
    d_model : int
        embed_dim.
    d_inner : int
        dff.
    dropout : float
        dropout rate.

    """

    def __init__(self, d, d_inner):
        super().__init__()
        self.w_1 = nn.Conv1d(d, d_inner, 1)
        self.w_2 = nn.Conv1d(d_inner, d, 1)

    def forward(self, x):
        """

        Parameters
        ----------
        x : `torch.Tensor`
            Tensor of shape (batch, len, embed_dim)

        """
        output = x.transpose(1, 2)
        output = self.w_2(F.relu(self.w_1(output)))
        output = output.transpose(1, 2)
        return output


def get_inputs():
    return [torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'d': 4, 'd_inner': 4}]
