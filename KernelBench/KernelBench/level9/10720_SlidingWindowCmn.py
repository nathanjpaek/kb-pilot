import torch
from torch import Tensor
import torchaudio.functional as F


class SlidingWindowCmn(torch.nn.Module):
    """
    Apply sliding-window cepstral mean (and optionally variance) normalization per utterance.

    Args:
        cmn_window (int, optional): Window in frames for running average CMN computation (int, default = 600)
        min_cmn_window (int, optional):  Minimum CMN window used at start of decoding (adds latency only at start).
            Only applicable if center == false, ignored if center==true (int, default = 100)
        center (bool, optional): If true, use a window centered on the current frame
            (to the extent possible, modulo end effects). If false, window is to the left. (bool, default = false)
        norm_vars (bool, optional): If true, normalize variance to one. (bool, default = false)
    """

    def __init__(self, cmn_window: 'int'=600, min_cmn_window: 'int'=100,
        center: 'bool'=False, norm_vars: 'bool'=False) ->None:
        super().__init__()
        self.cmn_window = cmn_window
        self.min_cmn_window = min_cmn_window
        self.center = center
        self.norm_vars = norm_vars

    def forward(self, waveform: 'Tensor') ->Tensor:
        """
        Args:
            waveform (Tensor): Tensor of audio of dimension (..., time).

        Returns:
            Tensor: Tensor of audio of dimension (..., time).
        """
        cmn_waveform = F.sliding_window_cmn(waveform, self.cmn_window, self
            .min_cmn_window, self.center, self.norm_vars)
        return cmn_waveform


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
