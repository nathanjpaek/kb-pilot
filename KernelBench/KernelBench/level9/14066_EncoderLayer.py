import torch
from typing import Tuple
from torch import nn
from typing import Optional


class ScaledDotProductAttention(nn.Module):
    """
    Scaled Dot-Product Attention

    Parameters
    ----------
    scale : float
        Scale factor (sqrt(d_k))

    dropout : float
        Dropout
    """

    def __init__(self, scale: 'float', dropout: 'float'=0.5) ->None:
        super(ScaledDotProductAttention, self).__init__()
        self.scale = scale
        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, Q: 'torch.Tensor', K: 'torch.Tensor', V:
        'torch.Tensor', mask: 'Optional[torch.Tensor]'=None):
        """
        Parameters
        ----------
        Q : torch.Tensor (batch_size, n_heads, word_pad_len, d_k)
            Query

        K : torch.Tensor
            Key

        V : torch.Tensor
            Value

        mask : torch.Tensor (batch_size, 1, 1, word_pad_len)
            Padding mask metrix, None if it is not needed

        Returns
        -------
        context : torch.Tensor (batch_size, n_heads, word_pad_len, d_k)
            Context vector

        att : torch.Tensor (batch_size, n_heads, word_pad_len, word_pad_len)
            Attention weights
        """
        att = torch.matmul(Q / self.scale, K.transpose(2, 3))
        if mask is not None:
            att = att.masked_fill(mask == 0, -1000000000.0)
        att = self.dropout(self.softmax(att))
        context = torch.matmul(att, V)
        return context, att


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Self-Attention

    Parameters
    ----------
    d_model : int
        Size of word embeddings

    n_heads : int
        Number of attention heads

    dropout : float
        Dropout
    """

    def __init__(self, d_model: 'int', n_heads: 'int', dropout: 'float'=0.5
        ) ->None:
        super(MultiHeadAttention, self).__init__()
        assert d_model % n_heads == 0
        self.d_k = d_model // n_heads
        self.n_heads = n_heads
        self.W_Q = nn.Linear(d_model, n_heads * self.d_k)
        self.W_K = nn.Linear(d_model, n_heads * self.d_k)
        self.W_V = nn.Linear(d_model, n_heads * self.d_k)
        scale = self.d_k ** 0.5
        self.attention = ScaledDotProductAttention(scale=scale)
        self.layer_norm = nn.LayerNorm(d_model)
        self.fc = nn.Linear(n_heads * self.d_k, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: 'torch.Tensor', mask: 'torch.Tensor') ->Tuple[
        torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        x : torch.Tensor (batch_size, word_pad_len, d_model)
            Input data

        mask : torch.Tensor (batch_size, 1, word_pad_len)
            Padding mask metrix, None if it is not needed

        Returns
        -------
        out : torch.Tensor (batch_size, word_pad_len, d_model)
            Output of multi-head self-attention network

        att: torch.Tensor (batch_size, n_heads, word_pad_len, word_pad_len)
            Attention weights
        """
        batch_size = x.size(0)
        Q = self.W_Q(x)
        K = self.W_K(x)
        V = self.W_V(x)
        Q = Q.view(batch_size, -1, self.n_heads, self.d_k)
        K = K.view(batch_size, -1, self.n_heads, self.d_k)
        V = V.view(batch_size, -1, self.n_heads, self.d_k)
        Q, K, V = Q.transpose(1, 2), K.transpose(1, 2), V.transpose(1, 2)
        if mask is not None:
            mask = mask.unsqueeze(1)
        context, att = self.attention(Q, K, V, mask=mask)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1,
            self.d_k * self.n_heads)
        out = self.dropout(self.fc(context))
        out = out + x
        out = self.layer_norm(out)
        return out, att


class PositionWiseFeedForward(nn.Module):
    """
    Position-Wise Feed-Forward Network

    Parameters
    ----------
    d_model : int
        Size of word embeddings

    hidden_size : int
        Size of position-wise feed forward network

    dropout : float
        Dropout
    """

    def __init__(self, d_model: 'int', hidden_size: 'int', dropout: 'float'=0.5
        ) ->None:
        super(PositionWiseFeedForward, self).__init__()
        self.W_1 = nn.Linear(d_model, hidden_size)
        self.W_2 = nn.Linear(hidden_size, d_model)
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x: 'torch.Tensor') ->torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor (batch_size, word_pad_len, d_model)
            Output of multi-head self-attention network

        Returns
        -------
        out : torch.Tensor (batch_size, word_pad_len, d_model)
            Output of position-wise feed-forward network
        """
        out = self.W_2(self.relu(self.W_1(x)))
        out = self.dropout(out)
        out += x
        out = self.layer_norm(out)
        return out


class EncoderLayer(nn.Module):
    """
    An encoder layer.

    Parameters
    ----------
    d_model : int
        Size of word embeddings

    n_heads : int
        Number of attention heads

    hidden_size : int
        Size of position-wise feed forward network

    dropout : float
        Dropout
    """

    def __init__(self, d_model: 'int', n_heads: 'int', hidden_size: 'int',
        dropout: 'float'=0.5) ->None:
        super(EncoderLayer, self).__init__()
        self.attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = PositionWiseFeedForward(d_model, hidden_size,
            dropout)

    def forward(self, x: 'torch.Tensor', mask: 'Optional[torch.Tensor]'=None
        ) ->Tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        x : torch.Tensor (batch_size, word_pad_len, d_model)
            Input data

        mask : torch.Tensor (batch_size, 1, word_pad_len)
            Padding mask metrix, None if it is not needed

        Returns
        -------
        out : torch.Tensor (batch_size, word_pad_len, d_model)
            Output of the current encoder layer

        att : torch.Tensor (batch_size, n_heads, word_pad_len, word_pad_len)
            Attention weights
        """
        att_out, att = self.attention(x, mask=mask)
        out = self.feed_forward(att_out)
        return out, att


def get_inputs():
    return [torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'d_model': 4, 'n_heads': 4, 'hidden_size': 4}]
