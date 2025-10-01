import torch
from torch import nn


def mask_(matrices, maskval=0.0, mask_diagonal=True):
    """
    Masks out all values in the given batch of matrices where i <= j holds,
    i < j if mask_diagonal is false

    In place operation

    :param tns:
    :return:
    """
    h, w = matrices.size(-2), matrices.size(-1)
    indices = torch.triu_indices(h, w, offset=0 if mask_diagonal else 1)
    matrices[..., indices[0], indices[1]] = maskval


class SelfAttentionGPT2(nn.Module):
    """
    This is the self-attention operation as implemented in the Huggingface port of GPT2. The code has been
    simplified to remove several features not used here but otherwise it should do exactly the same as GPT2 when run with
    normal parameters.

    It is very similar to the default SelfAttention below, with the exception of the way it's initialized and some
    small speed improvements in the custom implementation of the linear layer (the Conv1D defined above).

    We include this primarily for comparison with our own canonical implementation to check for performance differences.
    """

    def __init__(self, emb, heads, mask=False):
        super().__init__()
        self.nheads = heads
        self.emb = emb
        self.mask = mask
        self.c_attn = nn.Linear(emb, 3 * emb)
        self.c_proj = nn.Linear(emb, emb)

    def _attn(self, q, k, v):
        dot = torch.matmul(q, k)
        dot = dot / float(v.size(-1)) ** 0.5
        if self.mask:
            mask_(dot, maskval=float('-inf'), mask_diagonal=False)
        dot = nn.Softmax(dim=-1)(dot)
        return torch.matmul(dot, v)

    def merge_heads(self, x):
        x = x.permute(0, 2, 1, 3).contiguous()
        new_x_shape = x.size()[:-2] + (x.size(-2) * x.size(-1),)
        return x.view(*new_x_shape)

    def split_heads(self, x, is_key=False):
        new_x_shape = x.size()[:-1] + (self.nheads, x.size(-1) // self.nheads)
        x = x.view(*new_x_shape)
        if is_key:
            return x.permute(0, 2, 3, 1)
        else:
            return x.permute(0, 2, 1, 3)

    def forward(self, input_sequence):
        _b, _t, e = input_sequence.size()
        query, key, value = self.c_attn(input_sequence).split(e, dim=2)
        query = self.split_heads(query)
        key = self.split_heads(key, is_key=True)
        value = self.split_heads(value)
        a = self._attn(query, key, value)
        a = self.merge_heads(a)
        a = self.c_proj(a)
        return a


def get_inputs():
    return [torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'emb': 4, 'heads': 4}]
