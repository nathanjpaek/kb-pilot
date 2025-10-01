import math
import torch
import torch.nn as nn


class SimpleEmbed(nn.Module):

    def __init__(self, d_feat, embed_dim):
        super(SimpleEmbed, self).__init__()
        self.d_feat = d_feat
        self.embed_dim = embed_dim
        self.proj = nn.Linear(d_feat, embed_dim)

    def forward(self, x):
        x = x.reshape(len(x), self.d_feat, -1)
        x = x.permute(0, 2, 1)
        out = self.proj(x) * math.sqrt(self.embed_dim)
        return out


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'d_feat': 4, 'embed_dim': 4}]
