import torch
from torch import nn


class PatchEmbedding(nn.Module):
    """Image to Patch Embedding
    """

    def __init__(self, patch_size=16, embed_dim=768):
        super().__init__()
        self.proj = nn.Conv2d(3, embed_dim, patch_size, patch_size)

    def forward(self, x: 'torch.Tensor'):
        x = self.proj(x)
        x = x.permute(0, 2, 3, 1)
        return x


def get_inputs():
    return [torch.rand([4, 3, 64, 64])]


def get_init_inputs():
    return [[], {}]
