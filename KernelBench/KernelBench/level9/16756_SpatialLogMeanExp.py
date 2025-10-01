import torch
import numpy as np
import torch.nn as nn


def _lme(x, alpha, dim=-1, keepdim=False):
    """
    Apply log-mean-exp pooling with sharpness `alpha` across dimension `dim`.
    """
    if x.shape[dim] <= 1:
        return x if keepdim else x.squeeze(dim)
    if not torch.is_tensor(alpha) and alpha == 0:
        return x.mean(dim, keepdim=keepdim)
    if torch.is_tensor(alpha) or alpha != 1:
        x = x * alpha
    xmax, _ = x.max(dim=dim, keepdim=True)
    x = x - xmax
    x = torch.log(torch.mean(torch.exp(x), dim, keepdim=keepdim))
    if not keepdim:
        xmax = xmax.squeeze(dim)
        if torch.is_tensor(alpha) and abs(dim) <= alpha.dim():
            alpha = alpha.squeeze(dim)
    x = x + xmax
    if torch.is_tensor(alpha) or alpha != 1:
        x = x / alpha
    return x


class SpatialLogMeanExp(nn.Module):
    """
    Performs global log-mean-exp pooling over all spatial dimensions. If
    `trainable`, then the `sharpness` becomes a trainable parameter. If
    `per_channel`, then separate parameters are learned for the feature
    dimension (requires `in_channels`). If `exp`, the exponential of the
    trainable parameter is taken in the forward pass (i.e., the logarithm of
    the sharpness is learned). `per_channel`, `in_channels`, and `exp` are
    ignored if not `trainable`. If `keepdim`, will keep singleton spatial dims.
    See https://arxiv.org/abs/1411.6228, Eq. 6.
    """

    def __init__(self, sharpness=1, trainable=False, per_channel=False,
        in_channels=None, exp=False, keepdim=False):
        super(SpatialLogMeanExp, self).__init__()
        self.trainable = trainable
        if trainable:
            if exp:
                sharpness = np.log(sharpness)
            self.exp = exp
            if per_channel:
                if in_channels is None:
                    raise ValueError('per_channel requires in_channels')
                sharpness = torch.full((in_channels,), sharpness)
            else:
                sharpness = torch.tensor(sharpness)
            self.per_channel = per_channel
            sharpness = nn.Parameter(sharpness)
        self.sharpness = sharpness
        self.keepdim = keepdim

    def extra_repr(self):
        if not self.trainable:
            return 'sharpness={:.3g}, trainable=False'.format(self.sharpness)
        else:
            return 'trainable=True, per_channel={!r}, exp={!r}'.format(self
                .per_channel, self.exp)

    def forward(self, x):
        sharpness = self.sharpness
        if self.keepdim:
            spatial_dims = x.dim() - 2
        x = x.reshape(x.shape[:2] + (-1,))
        if x.shape[-1] <= 1:
            x = x.squeeze(-1)
        else:
            if self.trainable and self.exp:
                sharpness = torch.exp(sharpness)
            if self.trainable and self.per_channel:
                sharpness = sharpness.view(sharpness.shape + (1,))
            x = _lme(x, sharpness)
        if self.keepdim:
            x = x[(Ellipsis,) + (None,) * spatial_dims]
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
