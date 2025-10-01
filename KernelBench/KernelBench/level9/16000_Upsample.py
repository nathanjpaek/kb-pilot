import torch
import torch.nn as nn


class Upsample(nn.Module):
    """PyTorch upsampling implementation.

    This module upsamples by inserting <i-1> zeros in between samples in the time
    dimension.  It does not low pass filter after this and assumes that the filter is a
    separate module if desired.

    .. seealso:: rfml.ptradio.RRC

    Args:
        i (int, optional): Interpolation factor -- only integer valued sample rate
                           conversions are allowed. Defaults to 8.

    Raises:
        ValueError: If i is less than 2.

    This module assumes that the input is formatted as BxCxIQxT.  The time dimension is
    extended by a factor of *i*, as provided, and all other dimensions are untouched.
    """

    def __init__(self, i: 'int'=8):
        if i < 2:
            raise ValueError(
                'You must interpolate by at least a factor of 2, not {}'.
                format(i))
        super(Upsample, self).__init__()
        self.i = i

    def forward(self, x: 'torch.Tensor') ->torch.Tensor:
        x = x.unsqueeze(-1)
        mask = x.data.new(x.shape[0], x.shape[1], x.shape[2], x.shape[3],
            self.i).zero_()
        mask[:, :, :, :, 0] = 1.0
        x = mask * x
        x = x.reshape(x.shape[0], x.shape[1], x.shape[2], x.shape[3] * x.
            shape[4])
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
