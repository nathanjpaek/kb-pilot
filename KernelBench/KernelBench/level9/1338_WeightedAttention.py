import torch
from torch import nn


class WeightedAttention(nn.Module):

    def __init__(self, dim, eps=1e-08, softmax_dim=1, weighted_mean_dim=2):
        super().__init__()
        self.norm_input = nn.LayerNorm(dim)
        self.norm_context = nn.LayerNorm(dim)
        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(dim, dim)
        self.to_v = nn.Linear(dim, dim)
        self.eps = eps
        self.scale = dim ** -0.5
        self.softmax_dim = softmax_dim
        self.weighted_mean_dim = weighted_mean_dim

    def forward(self, inputs, context):
        inputs = self.norm_input(inputs)
        context = self.norm_context(context)
        q = self.to_q(inputs)
        k = self.to_k(context)
        v = self.to_v(context)
        dots = torch.einsum('bid,bjd->bij', q, k) * self.scale
        attn = dots.softmax(dim=self.softmax_dim) + self.eps
        attn = attn / attn.sum(dim=self.weighted_mean_dim, keepdim=True)
        updates = torch.einsum('bjd,bij->bid', v, attn)
        return updates


def get_inputs():
    return [torch.rand([4, 4, 4]), torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'dim': 4}]
