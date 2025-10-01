import torch
import torch.nn as nn
import torch.nn.functional as F


class ScaledDotProductAttention(nn.Module):
    """ Scaled Dot-Product Attention """

    def __init__(self, scale=None, attn_dropout=0.1):
        super().__init__()
        self.scale = scale
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):
        if self.scale is None:
            attn = torch.matmul(q / k.shape[-1] ** 0.5, k.transpose(2, 3))
        else:
            attn = torch.matmul(q / self.scale, k.transpose(2, 3))
        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1000000000.0)
        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)
        return output, attn


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand(
        [4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
