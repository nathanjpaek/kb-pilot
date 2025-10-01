import torch
import torch.utils.data
from itertools import chain as chain
import torch.nn as nn


class PatchEmbed3D(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, temporal_resolution=4, in_chans=3,
        patch_size=16, z_block_size=2, embed_dim=768, flatten=True):
        super().__init__()
        self.height = img_size // patch_size
        self.width = img_size // patch_size
        self.frames = temporal_resolution // z_block_size
        self.num_patches = self.height * self.width * self.frames
        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=(
            z_block_size, patch_size, patch_size), stride=(z_block_size,
            patch_size, patch_size))
        self.flatten = flatten

    def forward(self, x):
        _B, _C, _T, _H, _W = x.shape
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)
        return x


def get_inputs():
    return [torch.rand([4, 3, 64, 64, 64])]


def get_init_inputs():
    return [[], {}]
