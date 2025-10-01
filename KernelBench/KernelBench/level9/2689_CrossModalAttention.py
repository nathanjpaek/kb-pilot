import torch
from torch import nn


class CrossModalAttention(nn.Module):

    def __init__(self, emb_dim, num_heads, num_latents):
        super().__init__()
        self.value = nn.Parameter(torch.randn(num_latents, emb_dim))
        self.attention = nn.MultiheadAttention(emb_dim, num_heads)

    def forward(self, key, query):
        batch_size = key.shape[0]
        sa_value = self.value.unsqueeze(0).repeat(batch_size, 1, 1)
        attn_output, _attn_output_weights = self.attention(query, key, sa_value
            )
        return attn_output


def get_inputs():
    return [torch.rand([4, 4, 4]), torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'emb_dim': 4, 'num_heads': 4, 'num_latents': 4}]
