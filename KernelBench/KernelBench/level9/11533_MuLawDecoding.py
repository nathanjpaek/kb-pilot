import torch
from torch import Tensor
import torchaudio.functional as F


class MuLawDecoding(torch.nn.Module):
    """Decode mu-law encoded signal.  For more info see the
    `Wikipedia Entry <https://en.wikipedia.org/wiki/%CE%9C-law_algorithm>`_

    This expects an input with values between 0 and quantization_channels - 1
    and returns a signal scaled between -1 and 1.

    Args:
        quantization_channels (int, optional): Number of channels. (Default: ``256``)
    """
    __constants__ = ['quantization_channels']

    def __init__(self, quantization_channels: 'int'=256) ->None:
        super(MuLawDecoding, self).__init__()
        self.quantization_channels = quantization_channels

    def forward(self, x_mu: 'Tensor') ->Tensor:
        """
        Args:
            x_mu (Tensor): A mu-law encoded signal which needs to be decoded.

        Returns:
            Tensor: The signal decoded.
        """
        return F.mu_law_decoding(x_mu, self.quantization_channels)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
