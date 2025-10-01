import torch
import torch.nn.functional as F
from torch import nn


class AttentionPool2d(nn.Module):

    def __init__(self, spacial_dim: 'int', embed_dim: 'int', num_heads:
        'int', output_dim: 'int'=None):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim **
            2 + 1, embed_dim) / embed_dim ** 0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x):
        x = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3]).permute(
            2, 0, 1)
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)
        temp = self.positional_embedding[:, None, :]
        temp = temp.permute(2, 1, 0)
        temp = F.interpolate(temp, size=x.shape[0], mode='linear').permute(
            2, 1, 0)
        x = x + temp
        x, _ = F.multi_head_attention_forward(query=x, key=x, value=x,
            embed_dim_to_check=x.shape[-1], num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight, k_proj_weight=self.k_proj.
            weight, v_proj_weight=self.v_proj.weight, in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias,
            self.v_proj.bias]), bias_k=None, bias_v=None, add_zero_attn=
            False, dropout_p=0, out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias, use_separate_proj_weight=True,
            training=self.training, need_weights=False)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'spacial_dim': 4, 'embed_dim': 4, 'num_heads': 4}]
