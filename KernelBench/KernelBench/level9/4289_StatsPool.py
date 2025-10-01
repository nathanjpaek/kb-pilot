import torch
import warnings
import torch.nn as nn
from typing import Optional
import torch.optim
import torch.nn.functional as F


class StatsPool(nn.Module):
    """Statistics pooling

    Compute temporal mean and (unbiased) standard deviation
    and returns their concatenation.

    Reference
    ---------
    https://en.wikipedia.org/wiki/Weighted_arithmetic_mean

    """

    def forward(self, sequences: 'torch.Tensor', weights:
        'Optional[torch.Tensor]'=None) ->torch.Tensor:
        """Forward pass

        Parameters
        ----------
        sequences : (batch, channel, frames) torch.Tensor
            Sequences.
        weights : (batch, frames) torch.Tensor, optional
            When provided, compute weighted mean and standard deviation.

        Returns
        -------
        output : (batch, 2 * channel) torch.Tensor
            Concatenation of mean and (unbiased) standard deviation.
        """
        if weights is None:
            mean = sequences.mean(dim=2)
            std = sequences.std(dim=2, unbiased=True)
        else:
            weights = weights.unsqueeze(dim=1)
            num_frames = sequences.shape[2]
            num_weights = weights.shape[2]
            if num_frames != num_weights:
                warnings.warn(
                    f'Mismatch between frames ({num_frames}) and weights ({num_weights}) numbers.'
                    )
                weights = F.interpolate(weights, size=num_frames, mode=
                    'linear', align_corners=False)
            v1 = weights.sum(dim=2)
            mean = torch.sum(sequences * weights, dim=2) / v1
            dx2 = torch.square(sequences - mean.unsqueeze(2))
            v2 = torch.square(weights).sum(dim=2)
            var = torch.sum(dx2 * weights, dim=2) / (v1 - v2 / v1)
            std = torch.sqrt(var)
        return torch.cat([mean, std], dim=1)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
