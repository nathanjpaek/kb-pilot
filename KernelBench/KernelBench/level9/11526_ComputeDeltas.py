import torch
from torch import Tensor
import torchaudio.functional as F


class ComputeDeltas(torch.nn.Module):
    """Compute delta coefficients of a tensor, usually a spectrogram.

    See `torchaudio.functional.compute_deltas` for more details.

    Args:
        win_length (int): The window length used for computing delta. (Default: ``5``)
        mode (str): Mode parameter passed to padding. (Default: ``'replicate'``)
    """
    __constants__ = ['win_length']

    def __init__(self, win_length: 'int'=5, mode: 'str'='replicate') ->None:
        super(ComputeDeltas, self).__init__()
        self.win_length = win_length
        self.mode = mode

    def forward(self, specgram: 'Tensor') ->Tensor:
        """
        Args:
            specgram (Tensor): Tensor of audio of dimension (..., freq, time).

        Returns:
            Tensor: Tensor of deltas of dimension (..., freq, time).
        """
        return F.compute_deltas(specgram, win_length=self.win_length, mode=
            self.mode)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
