import torch
import torchvision.transforms.functional as F
import torch.nn as nn
import torch.nn.functional as F


class TransformerBlock(nn.Module):

    def __init__(self, seq_len: 'int', embed_channels: 'int', mlp_dims:
        'int', num_heads: 'int'):
        super().__init__()
        self.embed_channels = embed_channels
        self.seq_len = seq_len
        self.mlp_dims = mlp_dims
        self.num_heads = num_heads
        self.layer_norm = nn.LayerNorm([self.seq_len, self.embed_channels])
        self.self_attn = nn.MultiheadAttention(self.embed_channels, self.
            num_heads)
        self.emb_to_mlp = nn.Linear(self.embed_channels, self.mlp_dims)
        self.mlp_to_emb = nn.Linear(self.mlp_dims, self.embed_channels)

    def forward(self, x: 'torch.Tensor'):
        shortcut = x
        x = self.layer_norm(x)
        x, _ = self.self_attn(x, x, x)
        x = x + shortcut
        shortcut2 = x
        x = self.layer_norm(x)
        x = self.emb_to_mlp(x)
        x = F.gelu(x)
        x = self.mlp_to_emb(x)
        x = x + shortcut2
        return x


def get_inputs():
    return [torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'seq_len': 4, 'embed_channels': 4, 'mlp_dims': 4,
        'num_heads': 4}]
