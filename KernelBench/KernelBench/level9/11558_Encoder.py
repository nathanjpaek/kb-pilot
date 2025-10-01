import torch
from torch import nn
import torch.nn.functional as F


class Encoder(nn.Module):
    """Estimation of the nonnegative mixture weight by a 1-D conv layer.
    """

    def __init__(self, kernel_size, enc_dim, audio_channels):
        super(Encoder, self).__init__()
        self.conv1d_U = nn.Conv1d(audio_channels, enc_dim, kernel_size=
            kernel_size, stride=kernel_size // 2, bias=False)

    def forward(self, mixture):
        """
        Args:
            mixture: [M, T], M is batch size, T is #samples
        Returns:
            mixture_w: [M, N, K], where K = (T-L)/(L/2)+1 = 2T/L-1
        """
        mixture_w = F.relu(self.conv1d_U(mixture))
        return mixture_w


def get_inputs():
    return [torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'kernel_size': 4, 'enc_dim': 4, 'audio_channels': 4}]
