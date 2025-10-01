import math
import torch
from torch import nn


class ScaledDotProductAttention(nn.Module):
    """
    Scaled Dot-Product Attention Layer

    Attributes
    ----------
    softmax : nn.Functional
        softmax function applied at the last dimension
    """

    def __init__(self, dropout=0.1):
        super(ScaledDotProductAttention, self).__init__()
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, Q, K, V, mask=None):
        """
        Parameters
        ----------
        Q    : 4d tensor (batch_size, h, seq_len, d_K)
        K    : 4d tensor (batch_size, h, seq_len, d_K)
        V    : 4d tensor (batch_size, h, seq_len, d_V)
        mask : 2d tensor (seq_len, seq_len)
            2d binary tensor, where 1 means connection should be blocked and 
            0 means values can be fed forward to softmax
        
        Returns
        -------
        4d tensor (batch_size, h, seq_len, d_V)
        """
        scaled = torch.matmul(Q, K.transpose_(2, 3))
        scaled = scaled / math.sqrt(K.shape[3])
        if mask is not None:
            scaled.masked_fill_(mask, float('-inf'))
        scaled = self.softmax(scaled)
        scaled = self.dropout(scaled)
        attn = torch.matmul(scaled, V)
        return attn


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention Layer

    Attributes
    ----------
    h       : int
        number of parallel heads
    d_K     : int
        dimension of features in both query and key
    d_V     : int
        dimension of features in value
    d_model : int
        dimension of token embedding
    spd_attn: ScaledDotProductattention layer
        sub module to apply scaled dot product
    W_Q     : 2d tensor (d_model, d_K * h)
        learned parameters used to linearly project query to Q
    W_K     : 2d tensor (d_model, d_K * h)
        learned parameters used to linearly project key to K
    W_V     : 2d tensor (d_model, d_V * h)
        learned parameters used to linearly project val to V
    W_O     : 2d tensor (d_V * h, d_model)
        learned parameters used to linearly project scaled attention
        to the output tensor with the same dimension as input
    """

    def __init__(self, h, d_model, d_K, d_V, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        self.h = h
        self.d_K = d_K
        self.d_V = d_V
        self.d_model = d_model
        self.sdp_attn = ScaledDotProductAttention(dropout=dropout)
        self.W_Q = nn.Parameter(torch.Tensor(d_model, d_K * h))
        self.W_K = nn.Parameter(torch.Tensor(d_model, d_K * h))
        self.W_V = nn.Parameter(torch.Tensor(d_model, d_V * h))
        self.W_O = nn.Parameter(torch.Tensor(d_V * h, d_model))
        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self):
        slope = math.sqrt(5)
        nn.init.kaiming_uniform_(self.W_Q, a=slope)
        nn.init.kaiming_uniform_(self.W_K, a=slope)
        nn.init.kaiming_uniform_(self.W_V, a=slope)
        nn.init.kaiming_uniform_(self.W_O, a=slope)

    def forward(self, query, key, val, mask=None):
        """
        Parameters
        ----------
        query : 3d tensor (batch_size, seq_len, d_model)
            embedded query sequence
        key   : 3d tensor (batch_size, seq_len, d_model)
            embedded key sequence
        val   : 3d tensor (batch_size, seq_len, d_model)
            embedded value sequence
        mask  : 2d tensor (seq_len, seq_len)
            2d binary tensor, where 1 means pass, 0 means block

        Returns
        -------
        3d tensor (batch_size, seq_len, d_model)
        """
        h, d_K, d_V, _d_model = self.h, self.d_K, self.d_V, self.d_model
        W_Q, W_K, W_V, W_O = self.W_Q, self.W_K, self.W_V, self.W_O
        bs_q, l_q = query.shape[0], query.shape[1]
        bs_k, l_k = key.shape[0], key.shape[1]
        bs_v, l_v = val.shape[0], val.shape[1]
        Q = torch.matmul(query, W_Q)
        K = torch.matmul(key, W_K)
        V = torch.matmul(val, W_V)
        Q = Q.view(bs_q, l_q, h, d_K)
        K = K.view(bs_k, l_k, h, d_K)
        V = V.view(bs_v, l_v, h, d_V)
        Q.transpose_(1, 2)
        K.transpose_(1, 2)
        V.transpose_(1, 2)
        head = self.sdp_attn(Q, K, V, mask)
        head.transpose_(1, 2)
        head = head.reshape(bs_q, l_q, h * d_V)
        res = torch.matmul(head, W_O)
        return self.dropout(res)


class FeedForwardNetwork(nn.Module):
    """
    Position-wise Feed-Forward Network
    Apply the following operation
    f(x) = max(0, xW1 + b1) * W2 + b2

    Attributes
    ----------
    l1 : nn.Linear (in_features=d_model, out_features=d_ff)
    l2 : nn.Linear (in_features=d_ff, out_features=d_model)
    """

    def __init__(self, d_model, d_ff, dropout=0.1):
        """
        Parameters
        ----------
        d_model : int
            embedding length
        d_ff    : int
            size of intermediate activation
        """
        super(FeedForwardNetwork, self).__init__()
        self.l1 = nn.Linear(d_model, d_ff)
        self.l2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.l2(torch.relu(self.l1(x)))
        return self.dropout(x)


class EncoderUnit(nn.Module):
    """
    Encoder Unit - build block of encoder

    Attributes
    ----------
    attn  : MultiHeadAttention
    norm1 : LayerNorm
    ffn   : FeedForwardNetwork
    norm2 : LayerNorm
    """

    def __init__(self, h, d_model, d_K, d_V, d_ff, dropout=0.1):
        """
        Parameters
        ----------
        h       : int
            number of attention heads
        d_model : int
            size of embedding
        d_K     : int
            number of features in query and key
        d_V     : int
            number of features in value
        d_ff    : int
            dimension of the feed-forward network
        """
        super(EncoderUnit, self).__init__()
        self.attn = MultiHeadAttention(h, d_model, d_K, d_V)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = FeedForwardNetwork(d_model, d_ff)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, seq):
        """
        Parameters
        ----------
        seq : 3d tensor (batch_size, seq_len, d_model)
        
        Returns
        -------
        3d tensor (batch_size, seq_len, d_model)
        """
        a1 = self.attn(seq, seq, seq)
        a1 = self.norm1(seq + a1)
        a2 = self.ffn(a1)
        a2 = self.norm2(a1 + a2)
        return a2


def get_inputs():
    return [torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'h': 4, 'd_model': 4, 'd_K': 4, 'd_V': 4, 'd_ff': 4}]
