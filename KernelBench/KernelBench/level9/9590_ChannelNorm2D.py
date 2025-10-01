import torch
import torch.nn as nn


class ChannelNorm2D(nn.Module):
    """ 
    Similar to default Torch instanceNorm2D but calculates
    moments over channel dimension instead of spatial dims.
    Expects input_dim in format (B,C,H,W)
    """

    def __init__(self, input_channels, momentum=0.1, eps=0.001, affine=True,
        **kwargs):
        super(ChannelNorm2D, self).__init__()
        self.momentum = momentum
        self.eps = eps
        self.affine = affine
        if affine is True:
            self.gamma = nn.Parameter(torch.ones(1, input_channels, 1, 1))
            self.beta = nn.Parameter(torch.zeros(1, input_channels, 1, 1))

    def forward(self, x):
        """
        Calculate moments over channel dim, normalize.
        x:  Image tensor, shape (B,C,H,W)
        """
        mu, var = torch.mean(x, dim=1, keepdim=True), torch.var(x, dim=1,
            keepdim=True)
        x_normed = (x - mu) * torch.rsqrt(var + self.eps)
        if self.affine is True:
            x_normed = self.gamma * x_normed + self.beta
        return x_normed


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'input_channels': 4}]
