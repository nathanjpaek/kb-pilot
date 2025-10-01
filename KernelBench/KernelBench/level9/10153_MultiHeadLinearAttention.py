import torch
import torch.nn as nn
import torch.nn.functional as F


class Linear(nn.Linear):

    def __init__(self, in_features, out_features, bias=True):
        super(Linear, self).__init__(in_features=in_features, out_features=
            out_features, bias=bias)
        self.noise = None
        self.vn_std = None

    def init_vn(self, vn_std):
        self.vn_std = vn_std

    def sample_synaptic_noise(self, distributed):
        self.noise = torch.normal(mean=0.0, std=1.0, size=self.weight.size(
            ), device=self.weight.device, dtype=self.weight.dtype)
        if distributed:
            torch.distributed.broadcast(self.noise, 0)

    def forward(self, input):
        weight = self.weight
        if self.noise is not None and self.training:
            weight = weight + self.vn_std * self.noise
        return F.linear(input, weight, self.bias)


class MultiHeadAttention(nn.Module):
    """Mutli-Head Attention Layer

    Args:
        dim_model: model feature dimension
        num_heads: number of attention heads

    References: 
        Attention Is All You Need, Vaswani et al.
        https://arxiv.org/abs/1706.03762

    """

    def __init__(self, dim_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.dim_model = dim_model
        self.dim_head = dim_model // num_heads
        self.query_layer = Linear(self.dim_model, self.dim_model)
        self.key_layer = Linear(self.dim_model, self.dim_model)
        self.value_layer = Linear(self.dim_model, self.dim_model)
        self.output_layer = Linear(self.dim_model, self.dim_model)

    def forward(self, Q, K, V, mask=None):
        """Scaled Dot-Product Multi-Head Attention

        Args:
            Q: Query of shape (B, T, D)
            K: Key of shape (B, T, D)
            V: Value of shape (B, T, D)
            mask: Optional position mask of shape (1 or B, 1 or H, 1 or T, 1 or T)
        
        Return:
            O: Attention output of shape (B, T, D)
            att_w: Attention weights of shape (B, H, T, T)

        """
        batch_size = Q.size(0)
        Q = self.query_layer(Q)
        K = self.key_layer(K)
        V = self.value_layer(V)
        Q = Q.reshape(batch_size, -1, self.num_heads, self.dim_head).transpose(
            1, 2)
        K = K.reshape(batch_size, -1, self.num_heads, self.dim_head).transpose(
            1, 2)
        V = V.reshape(batch_size, -1, self.num_heads, self.dim_head).transpose(
            1, 2)
        att_scores = Q.matmul(K.transpose(2, 3)) / K.shape[-1] ** 0.5
        if mask is not None:
            att_scores += mask * -1000000000.0
        att_w = att_scores.softmax(dim=-1)
        O = att_w.matmul(V)
        O = O.transpose(1, 2).reshape(batch_size, -1, self.dim_model)
        O = self.output_layer(O)
        return O, att_w.detach()

    def pad(self, Q, K, V, mask, chunk_size):
        overflow_Q = Q.size(1) % chunk_size
        overflow_KV = K.size(1) % chunk_size
        padding_Q = chunk_size - overflow_Q if overflow_Q else 0
        padding_KV = chunk_size - overflow_KV if overflow_KV else 0
        batch_size, seq_len_KV, _ = K.size()
        Q = F.pad(Q, (0, 0, 0, padding_Q), value=0)
        K = F.pad(K, (0, 0, 0, padding_KV), value=0)
        V = F.pad(V, (0, 0, 0, padding_KV), value=0)
        if mask is not None:
            if mask.size(2) == 1:
                mask = F.pad(mask, pad=(0, padding_KV), value=1)
            else:
                mask = F.pad(mask, pad=(0, padding_Q, 0, padding_KV), value=1)
        elif padding_KV:
            mask = F.pad(Q.new_zeros(batch_size, 1, 1, seq_len_KV), pad=(0,
                padding_KV), value=1)
        return Q, K, V, mask, padding_Q


class MultiHeadLinearAttention(MultiHeadAttention):
    """Multi-Head Linear Attention

    Args:
        dim_model: model feature dimension
        num_heads: number of attention heads

    References: 
        Efficient Attention: Attention with Linear Complexities, Shen et al.
        https://arxiv.org/abs/1812.01243

        Efficient conformer-based speech recognition with linear attention, Li et al.
        https://arxiv.org/abs/2104.06865

    """

    def __init__(self, dim_model, num_heads):
        super(MultiHeadLinearAttention, self).__init__(dim_model, num_heads)

    def forward(self, Q, K, V):
        batch_size = Q.size(0)
        Q = self.query_layer(Q)
        K = self.key_layer(K)
        V = self.value_layer(V)
        Q = Q.reshape(batch_size, -1, self.num_heads, self.dim_head).transpose(
            1, 2)
        K = K.reshape(batch_size, -1, self.num_heads, self.dim_head).transpose(
            1, 2)
        V = V.reshape(batch_size, -1, self.num_heads, self.dim_head).transpose(
            1, 2)
        KV = (K / K.shape[-1] ** (1.0 / 4.0)).softmax(dim=-2).transpose(2, 3
            ).matmul(V)
        O = (Q / Q.shape[-1] ** (1.0 / 4.0)).softmax(dim=-1).matmul(KV)
        O = O.transpose(1, 2).reshape(batch_size, -1, self.dim_model)
        O = self.output_layer(O)
        return O, KV.detach()


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand(
        [4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'dim_model': 4, 'num_heads': 4}]
