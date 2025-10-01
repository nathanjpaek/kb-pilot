import torch
import torch.nn as nn
import torch.utils.data


class SpectralConvergence(nn.Module):

    def __init__(self):
        """Initilize spectral convergence loss module."""
        super().__init__()

    def forward(self, predicts_mag, targets_mag):
        """Calculate norm of difference operator.
    Args:
      predicts_mag (Tensor): Magnitude spectrogram of predicted signal (B, #frames, #freq_bins).
      targets_mag  (Tensor): Magnitude spectrogram of groundtruth signal (B, #frames, #freq_bins).
    Returns:
      Tensor: Spectral convergence loss value.
    """
        return torch.mean(torch.norm(targets_mag - predicts_mag, dim=(1, 2),
            p='fro') / torch.norm(targets_mag, dim=(1, 2), p='fro'))


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
