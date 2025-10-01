import torch
import torch.nn as nn


class PatchEmbedding(nn.Module):

    def __init__(self, image_size, patch_size, embed_dim, channels):
        super().__init__()
        self.image_size = image_size
        if image_size[0] % patch_size != 0 or image_size[1] % patch_size != 0:
            raise ValueError(
                'image dimensions must be divisible by the patch size')
        self.grid_size = image_size[0] // patch_size, image_size[1
            ] // patch_size
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.patch_size = patch_size
        self.proj = nn.Conv2d(channels, embed_dim, kernel_size=patch_size,
            stride=patch_size)

    def forward(self, im):
        _B, _C, _H, _W = im.shape
        x = self.proj(im).flatten(2).transpose(1, 2)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'image_size': [4, 4], 'patch_size': 4, 'embed_dim': 4,
        'channels': 4}]
