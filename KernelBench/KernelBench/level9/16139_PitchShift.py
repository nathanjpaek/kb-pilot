import torch
from torch import nn
import torch.nn.functional as F


class PitchShift(nn.Module):

    def __init__(self, shift):
        super(PitchShift, self).__init__()
        self.shift = shift

    def forward(self, x):
        if len(x.shape) == 2:
            x = x.unsqueeze(0)
        x = x.squeeze()
        mel_size = x.shape[1]
        shift_scale = (mel_size + self.shift) / mel_size
        x = F.interpolate(x.unsqueeze(1), scale_factor=(shift_scale, 1.0),
            align_corners=False, recompute_scale_factor=True, mode='bilinear'
            ).squeeze(1)
        x = x[:, :mel_size]
        if x.size(1) < mel_size:
            pad_size = mel_size - x.size(1)
            x = torch.cat([x, torch.zeros(x.size(0), pad_size, x.size(2))],
                dim=1)
        x = x.squeeze()
        return x.unsqueeze(1)


def get_inputs():
    return [torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'shift': 4}]
