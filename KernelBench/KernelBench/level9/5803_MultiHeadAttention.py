import torch
import numpy as np
import torch.utils.data
import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):

    def __init__(self, max_pos, d_k):
        super().__init__()
        self.w_rpr = nn.Linear(d_k, max_pos + 1, bias=False)

    def __call__(self, q, dist_matrices):
        return self.forward(q, dist_matrices)

    def forward(self, q, dist_matrices):
        """
        :param q:  [batch, heads, seq, d_k]
        :param dist_matrices:  list of dist_matrix , each of size n_edges X nedges
        :return: resampled_q_dot_rpr: [batch, heads, seq, seq]
        """
        q_dot_rpr = self.w_rpr(q)
        attn_rpr = self.resample_rpr_product(q_dot_rpr, dist_matrices)
        return attn_rpr

    @staticmethod
    def resample_rpr_product(q_dot_rpr, dist_matrices):
        """
        :param q_dot_rpr:  [batch, heads, seq, max_pos+1]
        :param dist_matrices: list of dist_matrix , each of size n_edges X nedges
        :return: resampled_q_dot_rpr: [batch, heads, seq, seq]
        """
        bs, _n_heads, max_seq, _ = q_dot_rpr.shape
        max_pos = q_dot_rpr.shape[-1] - 1
        seq_lens = np.array([d.shape[0] for d in dist_matrices])
        if (seq_lens == max_seq).all():
            pos_inds = np.stack(dist_matrices)
        else:
            pos_inds = np.ones((bs, max_seq, max_seq), dtype=np.int32
                ) * np.iinfo(np.int32).max
            for i_b in range(bs):
                dist_matrix = dist_matrices[i_b]
                n_edges = dist_matrix.shape[0]
                pos_inds[i_b, :n_edges, :n_edges] = dist_matrix
        pos_inds[pos_inds > max_pos] = max_pos
        batch_inds = np.arange(bs)[:, None, None]
        edge_inds = np.arange(max_seq)[None, :, None]
        resampled_q_dot_rpr = q_dot_rpr[batch_inds, :, edge_inds, pos_inds
            ].permute(0, 3, 1, 2)
        return resampled_q_dot_rpr


class PositionalScaledDotProductAttention(nn.Module):
    """ Scaled Dot-Product Attention with optional positional encodings """

    def __init__(self, temperature, positional_encoding=None, attn_dropout=0.1
        ):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.positional_encoding = positional_encoding

    def forward(self, q, k, v, mask=None, dist_matrices=None):
        """
        q: [batch, heads, seq, d_k]  queries
        k: [batch, heads, seq, d_k]  keys
        v: [batch, heads, seq, d_v]  values
        mask: [batch, 1, seq, seq]   for each edge, which other edges should be accounted for. "None" means all of them.
                                mask is important when using local attention, or when the meshes are of different sizes.
        rpr: [batch, seq, seq, d_k]  positional representations
        """
        attn_k = torch.matmul(q / self.temperature, k.transpose(2, 3))
        if self.positional_encoding is None:
            attn = attn_k
        else:
            attn_rpr = self.positional_encoding(q / self.temperature,
                dist_matrices)
            attn = attn_k + attn_rpr
        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1000000000.0)
        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)
        return output, attn


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention module
    from https://github.com/jadore801120/attention-is-all-you-need-pytorch
    by Yu-Hsiang Huang.
    use_values_as_is is our addition.
    """

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1,
        use_values_as_is=False, use_positional_encoding=False,
        max_relative_position=8):
        super().__init__()
        self.attention_type = type
        self.n_head = n_head
        self.d_k = d_k
        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        if not use_values_as_is:
            self.d_v = d_v
            self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
            self.fc = nn.Linear(n_head * d_v, d_model, bias=False)
        else:
            self.d_v = d_model
            self.w_vs = lambda x: self.__repeat_single_axis(x, -1, n_head)
            self.fc = lambda x: self.__average_head_results(x, n_head)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-06)
        positional_encoding = None
        if use_positional_encoding:
            positional_encoding = PositionalEncoding(max_relative_position, d_k
                )
        self.attention = PositionalScaledDotProductAttention(temperature=
            d_k ** 0.5, positional_encoding=positional_encoding)

    @staticmethod
    def __repeat_single_axis(x, axis, n_rep):
        rep_sizes = [1] * x.ndim
        rep_sizes[axis] = n_rep
        x_rep = x.repeat(rep_sizes)
        return x_rep

    @staticmethod
    def __average_head_results(x, n_head):
        shape = list(x.shape)[:-1] + [n_head, -1]
        avg_x = x.view(shape).mean(-2)
        return avg_x

    def forward(self, q, k, v, mask=None, dist_matrices=None):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)
        residual = q
        q = self.layer_norm(q)
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        if mask is not None:
            mask = mask.unsqueeze(1)
        q, attn = self.attention(q, k, v, mask=mask, dist_matrices=
            dist_matrices)
        q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        q = self.dropout(self.fc(q))
        q += residual
        return q, attn


def get_inputs():
    return [torch.rand([4, 4, 4]), torch.rand([4, 4, 4]), torch.rand([4, 4, 4])
        ]


def get_init_inputs():
    return [[], {'n_head': 4, 'd_model': 4, 'd_k': 4, 'd_v': 4}]
