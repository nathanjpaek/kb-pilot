import math
import torch
from torch import nn


def overlap_and_add(signal, frame_step):
    outer_dimensions = signal.size()[:-2]
    frames, frame_length = signal.size()[-2:]
    subframe_length = math.gcd(frame_length, frame_step)
    subframe_step = frame_step // subframe_length
    subframes_per_frame = frame_length // subframe_length
    output_size = frame_step * (frames - 1) + frame_length
    output_subframes = output_size // subframe_length
    subframe_signal = signal.view(*outer_dimensions, -1, subframe_length)
    frame = torch.arange(0, output_subframes, device=signal.device).unfold(
        0, subframes_per_frame, subframe_step)
    frame = frame.long()
    frame = frame.contiguous().view(-1)
    result = signal.new_zeros(*outer_dimensions, output_subframes,
        subframe_length)
    result.index_add_(-2, frame, subframe_signal)
    result = result.view(*outer_dimensions, -1)
    return result


class Decoder(nn.Module):

    def __init__(self, N, L, audio_channels):
        super(Decoder, self).__init__()
        self.N, self.L = N, L
        self.audio_channels = audio_channels
        self.basis_signals = nn.Linear(N, audio_channels * L, bias=False)
        """
        m = nn.Linear(20, 30)
        >>> input = torch.randn(128, 20)
        >>> output = m(input)
        >>> print(output.size())
        torch.Size([128, 30])
        """

    def forward(self, mixture_w, est_mask):
        """
        Args:
            mixture_w: [M, N, K]
            est_mask: [M, C, N, K]
        Returns:
            est_source: [M, C, T]
        """
        source_w = torch.unsqueeze(mixture_w, 1) * est_mask
        source_w = torch.transpose(source_w, 2, 3)
        est_source = self.basis_signals(source_w)
        m, c, k, _ = est_source.size()
        est_source = est_source.view(m, c, k, self.audio_channels, -1
            ).transpose(2, 3).contiguous()
        est_source = overlap_and_add(est_source, self.L // 2)
        return est_source


def get_inputs():
    return [torch.rand([4, 4, 4]), torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'N': 4, 'L': 4, 'audio_channels': 4}]
