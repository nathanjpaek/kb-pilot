from torch.nn import Module
import torch
from torch import nn
import torch.utils.data
import torch.nn.functional
import torch.autograd


class InstanceNorm(Module):
    """
    ## Instance Normalization Layer

    Instance normalization layer $\\text{IN}$ normalizes the input $X$ as follows:

    When input $X \\in \\mathbb{R}^{B \\times C \\times H \\times W}$ is a batch of image representations,
    where $B$ is the batch size, $C$ is the number of channels, $H$ is the height and $W$ is the width.
    $\\gamma \\in \\mathbb{R}^{C}$ and $\\beta \\in \\mathbb{R}^{C}$. The affine transformation with $gamma$ and
    $beta$ are optional.

    $$\\text{IN}(X) = \\gamma
    \\frac{X - \\underset{H, W}{\\mathbb{E}}[X]}{\\sqrt{\\underset{H, W}{Var}[X] + \\epsilon}}
    + \\beta$$
    """

    def __init__(self, channels: 'int', *, eps: float=1e-05, affine: bool=True
        ):
        """
        * `channels` is the number of features in the input
        * `eps` is $\\epsilon$, used in $\\sqrt{Var[X] + \\epsilon}$ for numerical stability
        * `affine` is whether to scale and shift the normalized value
        """
        super().__init__()
        self.channels = channels
        self.eps = eps
        self.affine = affine
        if self.affine:
            self.scale = nn.Parameter(torch.ones(channels))
            self.shift = nn.Parameter(torch.zeros(channels))

    def forward(self, x: 'torch.Tensor'):
        """
        `x` is a tensor of shape `[batch_size, channels, *]`.
        `*` denotes any number of (possibly 0) dimensions.
         For example, in an image (2D) convolution this will be
        `[batch_size, channels, height, width]`
        """
        x_shape = x.shape
        batch_size = x_shape[0]
        assert self.channels == x.shape[1]
        x = x.view(batch_size, self.channels, -1)
        mean = x.mean(dim=[-1], keepdim=True)
        mean_x2 = (x ** 2).mean(dim=[-1], keepdim=True)
        var = mean_x2 - mean ** 2
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        x_norm = x_norm.view(batch_size, self.channels, -1)
        if self.affine:
            x_norm = self.scale.view(1, -1, 1) * x_norm + self.shift.view(1,
                -1, 1)
        return x_norm.view(x_shape)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'channels': 4}]
