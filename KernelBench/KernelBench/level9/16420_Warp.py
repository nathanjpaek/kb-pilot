import torch
import torch.nn as nn


class Warp(torch.nn.Module):
    """Custom warping layer."""

    def __init__(self, mode='bilinear', padding_mode='reflection'):
        super().__init__()
        self.mode = mode
        self.padding_mode = padding_mode

    def forward(self, x, tform):
        """Warp the tensor `x` with `tform` along the time dimension.

        Parameters
        ----------
        x : torch.Tensor
            Tensor of shape `(n_samples, n_channels, lookback, n_assets)`.

        tform : torch.Tensor
            Tensor of shape `(n_samples, lookback)` or `(n_samples, lookback, n_assets)`.
            Note that in the first case the same transformation is going to be used over all
            assets. To prevent folding the transformation should be increasing along the
            time dimension. It should range from -1 (beginning of the series) to 1 (end of
            the series).

        Returns
        -------
        x_warped : torch.Tensor
            Warped version of input `x` with transformation `tform`. The shape is the same
            as the input shape - `(n_samples, n_channels, lookback, n_assets)`.

        """
        n_samples, _n_channels, lookback, n_assets = x.shape
        dtype, device = x.dtype, x.device
        if tform.ndim == 3:
            ty = tform
        elif tform.ndim == 2:
            ty = torch.stack(n_assets * [tform], dim=-1)
        else:
            raise ValueError(
                'The tform tensor needs to be either 2 or 3 dimensional.')
        tx = torch.ones(n_samples, lookback, n_assets, dtype=dtype, device=
            device)
        tx *= torch.linspace(-1, 1, steps=n_assets, device=device, dtype=dtype
            )[None, None, :]
        grid = torch.stack([tx, ty], dim=-1)
        x_warped = nn.functional.grid_sample(x, grid, mode=self.mode,
            padding_mode=self.padding_mode, align_corners=True)
        return x_warped


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4])]


def get_init_inputs():
    return [[], {}]
