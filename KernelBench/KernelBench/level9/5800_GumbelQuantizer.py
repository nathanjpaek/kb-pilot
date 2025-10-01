import torch
import torch.nn as nn
from torch.nn import functional as F


class GumbelQuantizer(nn.Module):

    def __init__(self, input_dim, num_latents, embedding_dim):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_latents = num_latents
        self.proj = nn.Conv2d(input_dim, num_latents, 1)
        self.embed = nn.Embedding(num_latents, embedding_dim)

    def forward(self, x):
        x = self.proj(x)
        soft_one_hot = F.gumbel_softmax(x, dim=1, hard=False, tau=1.0)
        z_q = torch.einsum('b n h w, n d -> b d h w', soft_one_hot, self.
            embed.weight)
        q_z = F.softmax(x, dim=1)
        kl = torch.sum(q_z * torch.log(q_z * self.num_latents + 1e-10), dim=1
            ).mean()
        return z_q, kl


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'input_dim': 4, 'num_latents': 4, 'embedding_dim': 4}]
