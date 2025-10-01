import torch
import torch.nn as nn
import torch.utils.data


class WeightNormConv2d(nn.Module):

    def __init__(self, in_dim, out_dim, kernel_size, stride=1, padding=0,
        bias=True, weight_norm=True, scale=False):
        """Intializes a Conv2d augmented with weight normalization.

        (See torch.nn.utils.weight_norm for detail.)

        Args:
            in_dim: number of input channels.
            out_dim: number of output channels.
            kernel_size: size of convolving kernel.
            stride: stride of convolution.
            padding: zero-padding added to both sides of input.
            bias: True if include learnable bias parameters, False otherwise.
            weight_norm: True if apply weight normalization, False otherwise.
            scale: True if include magnitude parameters, False otherwise.
        """
        super(WeightNormConv2d, self).__init__()
        if weight_norm:
            self.conv = nn.utils.weight_norm(nn.Conv2d(in_dim, out_dim,
                kernel_size, stride=stride, padding=padding, bias=bias))
            if not scale:
                self.conv.weight_g.data = torch.ones_like(self.conv.
                    weight_g.data)
                self.conv.weight_g.requires_grad = False
        else:
            self.conv = nn.Conv2d(in_dim, out_dim, kernel_size, stride=
                stride, padding=padding, bias=bias)

    def forward(self, x):
        """Forward pass.

        Args:
            x: input tensor.
        Returns:
            transformed tensor.
        """
        return self.conv(x)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_dim': 4, 'out_dim': 4, 'kernel_size': 4}]
