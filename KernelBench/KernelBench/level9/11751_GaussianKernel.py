import torch
import torch.nn as nn


class GaussianKernel(nn.Module):
    """
    Gaussian kernel module.

    :param mu: Float, mean of the kernel.
    :param sigma: Float, sigma of the kernel.

    Examples:
        >>> import torch
        >>> kernel = GaussianKernel()
        >>> x = torch.randn(4, 5, 10)
        >>> x.shape
        torch.Size([4, 5, 10])
        >>> kernel(x).shape
        torch.Size([4, 5, 10])

    """

    def __init__(self, mu: 'float'=1.0, sigma: 'float'=1.0):
        """Gaussian kernel constructor."""
        super().__init__()
        self.mu = mu
        self.sigma = sigma

    def forward(self, x):
        """Forward."""
        return torch.exp(-0.5 * (x - self.mu) ** 2 / self.sigma ** 2)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
