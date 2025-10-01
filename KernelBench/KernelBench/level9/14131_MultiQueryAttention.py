import torch
import torch.nn as nn
import torch.utils.cpp_extension


class MultiQueryAttention(nn.Module):

    def __init__(self, dim, latent_dim, num_heads):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.q = nn.Linear(dim, dim, bias=False)
        self.kv = nn.Linear(latent_dim, dim * 2, bias=False)
        self.o = nn.Linear(dim, dim, bias=False)
        self.scale = (dim // self.num_heads) ** -0.5

    def forward(self, x, z):
        B, xN, _ = x.size()
        _, zN, _ = z.size()
        Q = self.q(x).reshape(B, xN, self.num_heads, self.dim // self.num_heads
            ).transpose(1, 2)
        KV = self.kv(z).reshape(B, zN, 2, self.num_heads, self.dim // self.
            num_heads).permute(2, 0, 3, 1, 4)
        K, V = KV.unbind(dim=0)
        attn = Q @ K.transpose(-1, -2) * self.scale
        attn = attn.softmax(-1)
        O = (attn @ V).permute(0, 2, 1, 3).reshape(B, xN, self.dim)
        Z = self.o(O)
        return Z


def get_inputs():
    return [torch.rand([4, 4, 4]), torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'dim': 4, 'latent_dim': 4, 'num_heads': 4}]
