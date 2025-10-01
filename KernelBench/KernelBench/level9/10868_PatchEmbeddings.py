from torch.nn import Module
import torch
from torch import nn
import torch.utils.data
import torch.nn.functional
import torch.autograd


class PatchEmbeddings(Module):
    """
    <a id="PatchEmbeddings">
    ## Get patch embeddings
    </a>

    The paper splits the image into patches of equal size and do a linear transformation
    on the flattened pixels for each patch.

    We implement the same thing through a convolution layer, because it's simpler to implement.
    """

    def __init__(self, d_model: 'int', patch_size: 'int', in_channels: 'int'):
        """
        * `d_model` is the transformer embeddings size
        * `patch_size` is the size of the patch
        * `in_channels` is the number of channels in the input image (3 for rgb)
        """
        super().__init__()
        self.conv = nn.Conv2d(in_channels, d_model, patch_size, stride=
            patch_size)

    def forward(self, x: 'torch.Tensor'):
        """
        * `x` is the input image of shape `[batch_size, channels, height, width]`
        """
        x = self.conv(x)
        bs, c, h, w = x.shape
        x = x.permute(2, 3, 0, 1)
        x = x.view(h * w, bs, c)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'d_model': 4, 'patch_size': 4, 'in_channels': 4}]
