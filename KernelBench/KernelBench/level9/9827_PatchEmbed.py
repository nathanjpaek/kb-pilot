import torch
from torch import nn


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        num_patches_h = img_size[0] // patch_size
        num_patches_w = img_size[1] // patch_size
        num_patches = num_patches_h * num_patches_w
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.num_patches_h = num_patches_h
        self.num_patches_w = num_patches_w
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size,
            stride=patch_size)

    def forward(self, x):
        _B, _C, _H, _W = x.shape
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


def get_inputs():
    return [torch.rand([4, 3, 64, 64])]


def get_init_inputs():
    return [[], {'img_size': [4, 4]}]
