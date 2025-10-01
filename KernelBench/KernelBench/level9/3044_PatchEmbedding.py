import torch
import torch.nn as nn


class PatchEmbedding(nn.Module):
    """PatchEmdedding class
    Args:
        image_size(int): size of the image. assume that image shape is square
        in_channels(int): input channel of the image, 3 for RGB color channel
        embed_size(int): output channel size. This is the latent vector size.
                         and is constant throughout the transformer
        patch_size(int): size of the patch

    Attributes:
        n_patches(int): calculate the number of patches.
        patcher: convert image into patches. Basically a convolution layer with
                 kernel size and stride as of the patch size
    """

    def __init__(self, image_size=224, in_channels=3, embed_size=768,
        patch_size=16):
        super(PatchEmbedding, self).__init__()
        self.n_patches = (image_size // patch_size) ** 2
        self.patcher = nn.Conv2d(in_channels, embed_size, patch_size,
            patch_size)

    def forward(self, x):
        out = self.patcher(x)
        out = out.flatten(2)
        out = out.transpose(1, 2)
        return out


def get_inputs():
    return [torch.rand([4, 3, 64, 64])]


def get_init_inputs():
    return [[], {}]
