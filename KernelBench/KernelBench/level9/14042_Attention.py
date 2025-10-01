import torch
from typing import Tuple
from torch import nn


class Attention(nn.Module):
    """
    Attention network

    Parameters
    ----------
    rnn_size : int
        Size of Bi-LSTM
    """

    def __init__(self, rnn_size: 'int') ->None:
        super(Attention, self).__init__()
        self.w = nn.Linear(rnn_size, 1)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, H: 'torch.Tensor') ->Tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        H : torch.Tensor (batch_size, word_pad_len, hidden_size)
            Output of Bi-LSTM

        Returns
        -------
        r : torch.Tensor (batch_size, rnn_size)
            Sentence representation

        alpha : torch.Tensor (batch_size, word_pad_len)
            Attention weights
        """
        M = self.tanh(H)
        alpha = self.w(M).squeeze(2)
        alpha = self.softmax(alpha)
        r = H * alpha.unsqueeze(2)
        r = r.sum(dim=1)
        return r, alpha


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'rnn_size': 4}]
