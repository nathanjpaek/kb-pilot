import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.onnx


class MultiHeadAttention(nn.Module):

    def __init__(self, num_heads, emb_dim, dim_k=None, dropout=0.1):
        super().__init__()
        self.emb_dim = emb_dim
        self.dim_k = dim_k if dim_k else emb_dim // num_heads
        self.num_heads = num_heads
        self.q_linear = nn.Linear(emb_dim, self.dim_k * num_heads)
        self.k_linear = nn.Linear(emb_dim, self.dim_k * num_heads)
        self.v_linear = nn.Linear(emb_dim, self.dim_k * num_heads)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(self.dim_k * num_heads, emb_dim)

    def attention(self, q, k, v, dim_k, mask=None, dropout=None, explain=False
        ):
        k = k.transpose(-2, -1)
        if explain:
            None
        scores = torch.matmul(q, k) / math.sqrt(dim_k)
        if explain:
            None
        if mask is not None:
            mask = mask.unsqueeze(1)
            if explain:
                None
            scores = scores.masked_fill(mask == 0, -1000000000.0)
        softscores = F.softmax(scores, dim=-1)
        if dropout is not None:
            softscores = dropout(softscores)
        output = torch.matmul(softscores, v)
        return output, scores

    def forward(self, q, k, v, mask=None, explain=False):
        """
        inputs:
            q has shape (batch size, q_sequence length, embedding dimensions)
            k,v have shape (batch size, kv_sequence length, embedding dimensions)
            mask of shape (batch size, 1, kv_sequence length)
            explain: boolean, prints intermediate values if True
        outputs: sequence of vectors, re-represented using attention
            shape (batch size, q_sequence length, embedding dimensions)
        use:
            The encoder layer places the same source vector sequence into q,k,v
            and mask into mask.
            The decoder layer uses this twice, once with decoder inputs as q,k,v
            and target mask as mask. then with decoder inputs as q, encoder outputs
            as k, v and source mask as mask
        """
        batch_size = q.size(0)
        q = self.q_linear(q)
        k = self.k_linear(k)
        v = self.v_linear(v)
        if explain:
            None
        k = k.view(batch_size, -1, self.num_heads, self.dim_k)
        q = q.view(batch_size, -1, self.num_heads, self.dim_k)
        v = v.view(batch_size, -1, self.num_heads, self.dim_k)
        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)
        if explain:
            None
        attn, scores = self.attention(q, k, v, self.dim_k, mask, self.
            dropout, explain)
        if explain:
            None
        concat = attn.transpose(1, 2).contiguous().view(batch_size, -1, 
            self.dim_k * self.num_heads)
        if explain:
            None
        output = self.out(concat)
        if explain:
            None
        return output, scores


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand(
        [4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'num_heads': 4, 'emb_dim': 4}]
