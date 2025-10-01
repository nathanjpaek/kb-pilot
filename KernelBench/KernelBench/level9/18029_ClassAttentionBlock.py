import torch
from torch import Tensor
from torch import nn


class MLP(nn.Module):

    def __init__(self, dim, hidden_dim, out_dim=None) ->None:
        super().__init__()
        out_dim = out_dim or dim
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x: 'Tensor') ->Tensor:
        return self.fc2(self.act(self.fc1(x)))


class ClassAttention(nn.Module):
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
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads
            ).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        qc = q[:, :, 0:1]
        attn_cls = (qc * k).sum(dim=-1) * self.scale
        attn_cls = attn_cls.softmax(dim=-1)
        cls_token = (attn_cls.unsqueeze(2) @ v).transpose(1, 2).reshape(B, 1, C
            )
        cls_token = self.proj(cls_token)
        x = torch.cat([cls_token, x[:, 1:]], dim=1)
        return x


class ClassAttentionBlock(nn.Module):

    def __init__(self, dim, heads, eta=1e-05):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = ClassAttention(dim, heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, int(dim * 4))
        self.gamma1 = nn.Parameter(eta * torch.ones(dim))
        self.gamma2 = nn.Parameter(eta * torch.ones(dim))

    def forward(self, x: 'Tensor') ->Tensor:
        x = x + self.gamma1 * self.attn(self.norm1(x))
        x = self.norm2(x)
        x_res = x
        cls_token = self.gamma2 * self.mlp(x[:, :1])
        x = torch.cat([cls_token, x[:, 1:]], dim=1)
        x += x_res
        return x


def get_inputs():
    return [torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'dim': 4, 'heads': 4}]
