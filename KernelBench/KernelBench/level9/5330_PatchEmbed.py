import torch
import torch.nn as nn
from typing import Optional


class PatchEmbed(nn.Module):

    def __init__(self, img_size: 'int'=224, patch_size: 'int'=16, stride:
        'int'=None, in_channels: 'int'=3, embed_dim: 'int'=768, multi_conv:
        'bool'=False, norm_layer: 'Optional'=nn.LayerNorm):
        super(PatchEmbed, self).__init__()
        assert img_size % patch_size == 0, 'Argument `img_size` should be factor of argument `patch_size`'
        self.grid_size = img_size // patch_size
        self.patch_size = patch_size
        self.num_patches = self.grid_size ** 2
        if stride is None:
            stride = patch_size
        if multi_conv:
            if patch_size == 12:
                self.proj = nn.Sequential(nn.Conv2d(in_channels=in_channels,
                    out_channels=embed_dim // 4, kernel_size=7, stride=4,
                    padding=3), nn.ReLU(inplace=True), nn.Conv2d(
                    in_channels=embed_dim // 4, out_channels=embed_dim // 2,
                    kernel_size=3, stride=3), nn.ReLU(inplace=True), nn.
                    Conv2d(in_channels=embed_dim // 2, out_channels=
                    embed_dim, kernel_size=3, stride=1, padding=1))
            elif patch_size == 16:
                self.proj = nn.Sequential(nn.Conv2d(in_channels, embed_dim //
                    4, kernel_size=7, stride=4, padding=3), nn.ReLU(inplace
                    =True), nn.Conv2d(in_channels=embed_dim // 4,
                    out_channels=embed_dim // 2, kernel_size=3, stride=2,
                    padding=1), nn.ReLU(inplace=True), nn.Conv2d(
                    in_channels=embed_dim // 2, out_channels=embed_dim,
                    kernel_size=3, stride=2, padding=1))
        else:
            self.proj = nn.Conv2d(in_channels=in_channels, out_channels=
                embed_dim, kernel_size=patch_size, stride=stride)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = norm_layer

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(start_dim=2).transpose(1, 2)
        if self.norm is not None:
            x = self.norm(x)
        return x


def get_inputs():
    return [torch.rand([4, 3, 64, 64])]


def get_init_inputs():
    return [[], {}]
