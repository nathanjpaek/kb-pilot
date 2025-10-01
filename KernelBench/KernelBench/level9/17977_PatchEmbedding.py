import torch
from torch import Tensor
from torch import nn


class PatchEmbedding(nn.Module):
    """Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=16, embed_dim=768):
        super().__init__()
        assert img_size % patch_size == 0, 'Image size must be divisible by patch size'
        img_size = (img_size, img_size) if isinstance(img_size, int
            ) else img_size
        self.grid_size = img_size[0] // patch_size, img_size[1] // patch_size
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.proj = nn.Conv2d(3, embed_dim, patch_size, patch_size)

    def forward(self, x: 'Tensor') ->Tensor:
        x = self.proj(x)
        x = x.flatten(2).swapaxes(1, 2)
        return x


def get_inputs():
    return [torch.rand([4, 3, 64, 64])]


def get_init_inputs():
    return [[], {}]
