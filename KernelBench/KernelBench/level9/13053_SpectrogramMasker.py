import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectrogramMasker(nn.Module):
    """
    Helper class transforming wave-level mask to spectrogram-level mask
    """

    def __init__(self, win_length: 'int', hop_length: 'int'):
        super().__init__()
        self.win_length = win_length
        self.conv = nn.Conv1d(1, 1, self.win_length, stride=hop_length,
            padding=0, bias=False)
        torch.nn.init.constant_(self.conv.weight, 1.0 / self.win_length)

    def forward(self, wav_mask: 'torch.Tensor') ->torch.Tensor:
        with torch.no_grad():
            wav_mask = F.pad(wav_mask, [0, self.win_length // 2], value=0.0)
            wav_mask = F.pad(wav_mask, [self.win_length // 2, 0], value=1.0)
            mel_mask = self.conv(wav_mask.float().unsqueeze(1)).squeeze(1)
            mel_mask = torch.ceil(mel_mask)
        return mel_mask


def get_inputs():
    return [torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'win_length': 4, 'hop_length': 4}]
