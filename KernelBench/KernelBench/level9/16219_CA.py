import torch
from torch import Tensor
from torch import nn


class CA(nn.Module):
    """ClassAttention as in CaiT
    """

    def __init__(self, dim: 'int', heads: 'int'):
        super().__init__()
        self.num_heads = heads
        self.scale = (dim // heads) ** -0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: 'Tensor') ->Tensor:
        B, N, C = x.shape
        q, k, v = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.
            num_heads).permute(2, 0, 3, 1, 4)
        qc = q[:, :, :1]
        attn_cls = (qc * k).sum(dim=-1) * self.scale
        attn_cls = attn_cls.softmax(dim=-1)
        cls_token = (attn_cls.unsqueeze(2) @ v).transpose(1, 2).reshape(B, 1, C
            )
        cls_token = self.proj(cls_token)
        x = torch.cat([cls_token, x[:, 1:]], dim=1)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'dim': 4, 'heads': 4}]
