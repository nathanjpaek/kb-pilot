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


class PatchMerging(nn.Module):
    """
    This class implements the patch merging approach which is essential a strided convolution with normalization before
    """

    def __init__(self, in_channels: 'int') ->None:
        """
        Constructor method
        :param in_channels: (int) Number of input channels
        """
        super(PatchMerging, self).__init__()
        self.normalization: 'nn.Module' = nn.LayerNorm(normalized_shape=4 *
            in_channels)
        self.linear_mapping: 'nn.Module' = nn.Linear(in_features=4 *
            in_channels, out_features=2 * in_channels, bias=False)

    def forward(self, input: 'torch.Tensor') ->torch.Tensor:
        """
        Forward pass
        :param input: (torch.Tensor) Input tensor of the shape [batch size, in channels, height, width]
        :return: (torch.Tensor) Output tensor of the shape [batch size, 2 * in channels, height // 2, width // 2]
        """
        batch_size, _channels, _height, _width = input.shape
        input: 'torch.Tensor' = bchw_to_bhwc(input)
        input: 'torch.Tensor' = input.unfold(dimension=1, size=2, step=2
            ).unfold(dimension=2, size=2, step=2)
        input: 'torch.Tensor' = input.reshape(batch_size, input.shape[1],
            input.shape[2], -1)
        input: 'torch.Tensor' = self.normalization(input)
        output: 'torch.Tensor' = bhwc_to_bchw(self.linear_mapping(input))
        return output


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 4}]
