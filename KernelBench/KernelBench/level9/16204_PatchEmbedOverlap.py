import torch
from torch import Tensor
from torch import nn


class PatchEmbedOverlap(nn.Module):
    """Image to Patch Embedding with overlapping
    """

    def __init__(self, patch_size=16, stride=16, padding=0, embed_dim=768):
        super().__init__()
        self.proj = nn.Conv2d(3, embed_dim, patch_size, stride, padding)

    def forward(self, x: 'torch.Tensor') ->Tensor:
        x = self.proj(x)
        return x


def get_inputs():
    return [torch.rand([4, 3, 64, 64])]


def get_init_inputs():
    return [[], {}]
