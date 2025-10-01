import torch
from torch import nn


class PatchEmbed(nn.Module):

    def __init__(self, input_shape=[224, 224], patch_size=16, in_chans=3,
        num_features=768, norm_layer=None, flatten=True):
        super().__init__()
        self.num_patches = input_shape[0] // patch_size * (input_shape[1] //
            patch_size)
        self.flatten = flatten
        self.proj = nn.Conv2d(in_chans, num_features, kernel_size=
            patch_size, stride=patch_size)
        self.norm = norm_layer(num_features) if norm_layer else nn.Identity()

    def forward(self, x):
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x


def get_inputs():
    return [torch.rand([4, 3, 64, 64])]


def get_init_inputs():
    return [[], {}]
