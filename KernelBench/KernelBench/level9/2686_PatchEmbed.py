import torch
import torch.nn as nn
import torch.nn.parallel


class PatchEmbed(nn.Module):
    """
    Image to Patch Embedding.
    Different with ViT use 1 conv layer, we use 4 conv layers to do patch embedding
    """

    def __init__(self, img_size=224, stem_conv=False, stem_stride=1,
        patch_size=8, in_chans=3, hidden_dim=64, embed_dim=384):
        super().__init__()
        assert patch_size in [4, 8, 16]
        self.stem_conv = stem_conv
        if stem_conv:
            self.conv = nn.Sequential(nn.Conv2d(in_chans, hidden_dim,
                kernel_size=7, stride=stem_stride, padding=3, bias=False),
                nn.BatchNorm2d(hidden_dim), nn.ReLU(inplace=True), nn.
                Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=1,
                padding=1, bias=False), nn.BatchNorm2d(hidden_dim), nn.ReLU
                (inplace=True), nn.Conv2d(hidden_dim, hidden_dim,
                kernel_size=3, stride=1, padding=1, bias=False), nn.
                BatchNorm2d(hidden_dim), nn.ReLU(inplace=True))
        self.proj = nn.Conv2d(hidden_dim, embed_dim, kernel_size=patch_size //
            stem_stride, stride=patch_size // stem_stride)
        self.num_patches = img_size // patch_size * (img_size // patch_size)

    def forward(self, x):
        if self.stem_conv:
            x = self.conv(x)
        x = self.proj(x)
        return x


def get_inputs():
    return [torch.rand([4, 64, 64, 64])]


def get_init_inputs():
    return [[], {}]
