import torch


class UnpoolAvgHealpix(torch.nn.Module):
    """Healpix Average Unpooling module
    
    Parameters
    ----------
    kernel_size : int
        Pooling kernel width
    """

    def __init__(self, kernel_size, *args, **kwargs):
        """kernel_size should be 4, 16, 64, etc."""
        super().__init__()
        self.kernel_size = kernel_size

    def extra_repr(self):
        return 'kernel_size={kernel_size}'.format(**self.__dict__)

    def forward(self, x, *args):
        """x has shape (batch, pixels, channels) and is in nested ordering"""
        x = x.permute(0, 2, 1)
        x = torch.nn.functional.interpolate(x, scale_factor=self.
            kernel_size, mode='nearest')
        return x.permute(0, 2, 1)


def get_inputs():
    return [torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'kernel_size': 4}]
