import math
import torch
import numpy as np
import torch as th
import torch.nn as nn


def count_flops_attn(model, _x, y):
    """
    A counter for the `thop` package to count the operations in an
    attention operation.
    Meant to be used like:
        macs, params = thop.profile(
            model,
            inputs=(inputs, timestamps),
            custom_ops={QKVAttention: QKVAttention.count_flops},
        )
    """
    b, c, *spatial = y[0].shape
    num_spatial = int(np.prod(spatial))
    matmul_ops = 2 * b * num_spatial ** 2 * c
    model.total_ops += th.DoubleTensor([matmul_ops])


class QKVAttentionLegacy(nn.Module):
    """
    A module which performs QKV attention. Matches legacy QKVAttention + input/ouput heads shaping
    """

    def __init__(self, n_heads, dropout=0.0):
        super().__init__()
        self.n_heads = n_heads
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, qkv, y):
        """
        Apply QKV attention.

        :param qkv: an [N x (H * 3 * C) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x (H * C) x T] tensor after attention.
        """
        bs, width, length = qkv.shape
        if y is None:
            assert width % (3 * self.n_heads) == 0
            ch = width // (3 * self.n_heads)
            q, k, v = qkv.reshape(bs * self.n_heads, ch * 3, length).split(ch,
                dim=1)
        else:
            assert width % self.n_heads == 0
            ch = width // self.n_heads
            q = qkv.reshape(bs * self.n_heads, ch, length)
            k = v = y.reshape(bs * self.n_heads, ch, -1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = th.einsum('bct,bcs->bts', q * scale, k * scale)
        weight = self.dropout(th.softmax(weight.float(), dim=-1).type(
            weight.dtype))
        a = th.einsum('bts,bcs->bct', weight, v)
        return a.reshape(bs, -1, length)

    @staticmethod
    def count_flops(model, _x, y):
        return count_flops_attn(model, _x, y)


def get_inputs():
    return [torch.rand([4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'n_heads': 4}]
