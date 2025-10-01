import torch
import torch.nn as nn


def bchw_to_bhwc(input: 'torch.Tensor') ->torch.Tensor:
    """
    Permutes a tensor to the shape [batch size, height, width, channels]
    :param input: (torch.Tensor) Input tensor of the shape [batch size, height, width, channels]
    :return: (torch.Tensor) Output tensor of the shape [batch size, height, width, channels]
    """
    return input.permute(0, 2, 3, 1)


def bhwc_to_bchw(input: 'torch.Tensor') ->torch.Tensor:
    """
    Permutes a tensor to the shape [batch size, channels, height, width]
    :param input: (torch.Tensor) Input tensor of the shape [batch size, height, width, channels]
    :return: (torch.Tensor) Output tensor of the shape [batch size, channels, height, width]
    """
    return input.permute(0, 3, 1, 2)


class PatchEmbedding(nn.Module):
    """
    Module embeds a given image into patch embeddings.
    """

    def __init__(self, in_channels: 'int'=3, out_channels: 'int'=96,
        patch_size: 'int'=4) ->None:
        """
        Constructor method
        :param in_channels: (int) Number of input channels
        :param out_channels: (int) Number of output channels
        :param patch_size: (int) Patch size to be utilized
        :param image_size: (int) Image size to be used
        """
        super(PatchEmbedding, self).__init__()
        self.out_channels: 'int' = out_channels
        self.linear_embedding: 'nn.Module' = nn.Conv2d(in_channels=
            in_channels, out_channels=out_channels, kernel_size=(patch_size,
            patch_size), stride=(patch_size, patch_size))
        self.normalization: 'nn.Module' = nn.LayerNorm(normalized_shape=
            out_channels)

    def forward(self, input: 'torch.Tensor') ->torch.Tensor:
        """
        Forward pass transforms a given batch of images into a patch embedding
        :param input: (torch.Tensor) Input images of the shape [batch size, in channels, height, width]
        :return: (torch.Tensor) Patch embedding of the shape [batch size, patches + 1, out channels]
        """
        embedding: 'torch.Tensor' = self.linear_embedding(input)
        embedding: 'torch.Tensor' = bhwc_to_bchw(self.normalization(
            bchw_to_bhwc(embedding)))
        return embedding


def get_inputs():
    return [torch.rand([4, 3, 64, 64])]


def get_init_inputs():
    return [[], {}]
