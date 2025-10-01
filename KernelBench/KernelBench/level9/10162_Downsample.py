import torch
import torch.nn as nn
import torch.nn.parallel


class Downsample(nn.Module):
    """
    Image to Patch Embedding, downsampling between stage1 and stage2
    """

    def __init__(self, in_embed_dim, out_embed_dim, patch_size):
        super().__init__()
        self.proj = nn.Conv2d(in_embed_dim, out_embed_dim, kernel_size=
            patch_size, stride=patch_size)

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        x = self.proj(x)
        x = x.permute(0, 2, 3, 1)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_embed_dim': 4, 'out_embed_dim': 4, 'patch_size': 4}]
