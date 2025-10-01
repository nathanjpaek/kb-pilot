import torch
from torch import Tensor
from typing import Tuple
import torch.nn.functional as F
import torch.nn.functional


class Normalize(torch.nn.Module):
    """Normalize a tensor time series with mean and standard deviation.
    Given mean: ``(mean[1],...,mean[n])`` and std: ``(std[1],..,std[n])`` for ``n``
    channels, this transform will normalize each channel of the input
    ``torch.*Tensor`` i.e.,
    ``output[channel] = (input[channel] - mean[channel]) / std[channel]``

    .. note::
        This transform acts out of place, i.e., it does not mutate the input tensor.

    Args:
        mean (tuple): Sequence of means for each channel.
        std (tuple): Sequence of standard deviations for each channel.
        inplace(bool,optional): Bool to make this operation in-place.

    """

    def __init__(self, mean: 'Tuple[float]', std: 'Tuple[float]', inplace:
        'bool'=False):
        super(Normalize, self).__init__()
        self.mean = mean
        self.std = std
        self.inplace = inplace

    def forward(self, tensor: 'Tensor') ->Tensor:
        """
        Args:
            tensor (Tensor): Tensor time series to be normalized.

        Returns:
            Tensor: Normalized Tensor series.
        """
        return F.normalize(tensor, self.mean, self.std, self.inplace)

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.
            mean, self.std)


def get_inputs():
    return [torch.rand([4, 4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'mean': 4, 'std': 4}]
