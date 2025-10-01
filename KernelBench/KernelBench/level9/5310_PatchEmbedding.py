import torch
import torch.nn as nn


class PatchEmbedding(nn.Module):
    """
    small patches embedding
    image(B, C, H, W) -> projection(B, emb_dims, H/P, W/P) -> flatten & transpose(B, {(H/P) * (W/P)}, embed_dims)
    """

    def __init__(self, image_size=224, patch_size=16, in_channels=3,
        embed_dims=768, norm_layer=None, flatten=True):
        """
        Args:
            image_size (int, optional): input image size. Defaults to 224.
            patch_size (int, optional): patch image size. Defaults to 16.
            in_channels (int, optional): input image channels, almost 3. Defaults to 3.
            embed_dims (int, optional): patch embedding dimension. Defaults to 768.
            norm_layer (nn.Module, optional): if exists, it means LayerNorm. Defaults to None.
            flatten (bool, optional): flatten the last two layers. Defaults to True.
        """
        super(PatchEmbedding, self).__init__()
        self.image_size = image_size if isinstance(image_size, tuple) else (
            image_size, image_size)
        self.patch_size = patch_size if isinstance(patch_size, tuple) else (
            patch_size, patch_size)
        self.num_patches = self.image_size[0] // self.patch_size[0] * (self
            .image_size[1] // self.patch_size[1])
        self.flatten = flatten
        self.projection = nn.Conv2d(in_channels, embed_dims, kernel_size=
            self.patch_size, stride=self.patch_size)
        self.norm_layer = norm_layer(embed_dims
            ) if norm_layer else nn.Identity()

    def forward(self, x):
        x = self.projection(x)
        if self.flatten:
            x = torch.flatten(x, start_dim=2, end_dim=-1)
            x = torch.transpose(x, 1, 2)
        x = self.norm_layer(x)
        return x


def get_inputs():
    return [torch.rand([4, 3, 64, 64])]


def get_init_inputs():
    return [[], {}]
