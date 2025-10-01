import torch
from typing import *


class RotaryEmbedding(torch.nn.Module):
    """`Rotary Position Embedding <https://arxiv.org/abs/2104.09864v2>

    Args:
        rotary_dim (int): rotary dimension
    """

    def __init__(self, rotary_dim: 'int'):
        super().__init__()
        self.rotary_dim = rotary_dim

    def fixed_pos_embedding(self, x, seq_len=None, dtype=torch.float):
        dim = x.shape[-1]
        inv_freq = 1.0 / 10000 ** (torch.arange(0, dim, 2) / dim)
        sinusoid_inp = torch.einsum('i , j -> i j', torch.arange(seq_len),
            inv_freq).to(x.device)
        return torch.sin(sinusoid_inp), torch.cos(sinusoid_inp)

    def rotate_every_two(self, x):
        if x.dim() == 4:
            x1 = x[:, :, :, ::2]
            x2 = x[:, :, :, 1::2]
        else:
            x1 = x[:, :, ::2]
            x2 = x[:, :, 1::2]
        x = torch.stack((-x2, x1), axis=-1)
        return x.flatten(-2)

    def apply_rotary_pos_emb(self, x, sincos, offset=0):
        sin, cos = map(lambda t: t[None, offset:x.shape[-2] + offset, :].
            repeat_interleave(2, 2), sincos)
        return x * cos + self.rotate_every_two(x) * sin

    def forward(self, h_q, h_k):
        """
        Args:
            h_q : (batch_size, num_head, len_q, dim_head)
            h_k : (batch_size, k_num_head, len_k, dim_head)

        Return:
            h_q : (batch_size, num_head, len_q, dim_head)
            h_k : (batch_size, k_num_head, len_k, dim_head)
        """
        if h_q.dim() == 4:
            q_rot = h_q[:, :, :, :self.rotary_dim]
            q_pass = h_q[:, :, :, self.rotary_dim:]
            k_rot = h_k[:, :, :, :self.rotary_dim]
            k_pass = h_k[:, :, :, self.rotary_dim:]
        else:
            q_rot = h_q[:, :, :self.rotary_dim]
            q_pass = h_q[:, :, self.rotary_dim:]
            k_rot = h_k[:, :, :self.rotary_dim]
            k_pass = h_k[:, :, self.rotary_dim:]
        seq_len = h_k.shape[-2]
        sincos = self.fixed_pos_embedding(k_rot, seq_len=seq_len, dtype=h_k
            .dtype)
        k_rot = self.apply_rotary_pos_emb(k_rot, sincos, offset=0)
        q_rot = self.apply_rotary_pos_emb(q_rot, sincos, offset=0)
        h_q = torch.cat([q_rot, q_pass], dim=-1)
        h_k = torch.cat([k_rot, k_pass], dim=-1)
        return h_q, h_k


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'rotary_dim': 4}]
