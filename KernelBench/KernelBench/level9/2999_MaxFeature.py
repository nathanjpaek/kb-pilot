import torch
import torch.nn as nn


class MaxFeature(nn.Module):
    """Conv2d or Linear layer with max feature selector

    Generate feature maps with double channels, split them and select the max
        feature.

    Args:
        in_channels (int): Channel number of inputs.
        out_channels (int): Channel number of outputs.
        kernel_size (int or tuple): Size of the convolving kernel.
        stride (int or tuple, optional): Stride of the convolution. Default: 1
        padding (int or tuple, optional): Zero-padding added to both sides of
            the input. Default: 1
        filter_type (str): Type of filter. Options are 'conv2d' and 'linear'.
            Default: 'conv2d'.
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
        padding=1, filter_type='conv2d'):
        super().__init__()
        self.out_channels = out_channels
        filter_type = filter_type.lower()
        if filter_type == 'conv2d':
            self.filter = nn.Conv2d(in_channels, 2 * out_channels,
                kernel_size=kernel_size, stride=stride, padding=padding)
        elif filter_type == 'linear':
            self.filter = nn.Linear(in_channels, 2 * out_channels)
        else:
            raise ValueError(
                f"'filter_type' should be 'conv2d' or 'linear', but got {filter_type}"
                )

    def forward(self, x):
        """Forward function.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Forward results.
        """
        x = self.filter(x)
        out = torch.chunk(x, chunks=2, dim=1)
        return torch.max(out[0], out[1])


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 4, 'out_channels': 4}]
