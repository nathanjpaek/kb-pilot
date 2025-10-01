import torch
from torch import nn
from torch import einsum


class SpatialAttention(nn.Module):

    def __init__(self):
        super().__init__()

    def similarity(self, spatial_embedding):
        e0 = spatial_embedding.unsqueeze(2)
        e1 = spatial_embedding.unsqueeze(1)
        dist = (e0 - e1).norm(2, dim=-1)
        sim = (-dist.pow(2)).exp()
        sim = sim / sim.sum(dim=-1, keepdims=True)
        return sim

    def forward(self, spatial_embedding, z):
        attn = self.similarity(spatial_embedding)
        out = einsum('b i j, b j d -> b i d', attn, z)
        return out


def get_inputs():
    return [torch.rand([4, 4, 4]), torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {}]
