import torch
from torch import Tensor
import torchaudio.functional as F


class MuLawEncoding(torch.nn.Module):
    """Encode signal based on mu-law companding.  For more info see the
    `Wikipedia Entry <https://en.wikipedia.org/wiki/%CE%9C-law_algorithm>`_

    This algorithm assumes the signal has been scaled to between -1 and 1 and
    returns a signal encoded with values from 0 to quantization_channels - 1

    Args:
        quantization_channels (int, optional): Number of channels. (Default: ``256``)
    """
    __constants__ = ['quantization_channels']

    def __init__(self, quantization_channels: 'int'=256) ->None:
        super(MuLawEncoding, self).__init__()
        self.quantization_channels = quantization_channels

    def forward(self, x: 'Tensor') ->Tensor:
        """
        Args:
            x (Tensor): A signal to be encoded.

        Returns:
            x_mu (Tensor): An encoded signal.
        """
        return F.mu_law_encoding(x, self.quantization_channels)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
