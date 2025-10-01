import math
import torch
from torch import Tensor
import torchaudio.functional as F


class Vol(torch.nn.Module):
    """Add a volume to an waveform.

    Args:
        gain (float): Interpreted according to the given gain_type:
            If ``gain_type`` = ``amplitude``, ``gain`` is a positive amplitude ratio.
            If ``gain_type`` = ``power``, ``gain`` is a power (voltage squared).
            If ``gain_type`` = ``db``, ``gain`` is in decibels.
        gain_type (str, optional): Type of gain. One of: ``amplitude``, ``power``, ``db`` (Default: ``amplitude``)
    """

    def __init__(self, gain: 'float', gain_type: 'str'='amplitude'):
        super(Vol, self).__init__()
        self.gain = gain
        self.gain_type = gain_type
        if gain_type in ['amplitude', 'power'] and gain < 0:
            raise ValueError(
                'If gain_type = amplitude or power, gain must be positive.')

    def forward(self, waveform: 'Tensor') ->Tensor:
        """
        Args:
            waveform (Tensor): Tensor of audio of dimension (..., time).

        Returns:
            Tensor: Tensor of audio of dimension (..., time).
        """
        if self.gain_type == 'amplitude':
            waveform = waveform * self.gain
        if self.gain_type == 'db':
            waveform = F.gain(waveform, self.gain)
        if self.gain_type == 'power':
            waveform = F.gain(waveform, 10 * math.log10(self.gain))
        return torch.clamp(waveform, -1, 1)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'gain': 4}]
