import torch
import torchaudio


class DoubleDeltaTransform(torch.nn.Module):
    """A transformation to compute delta and double delta features.

    Args:
        win_length (int): The window length to use for computing deltas (Default: 5).
        mode (str): Mode parameter passed to padding (Default: replicate).
    """

    def __init__(self, win_length: 'int'=5, mode: 'str'='replicate') ->None:
        super().__init__()
        self.win_length = win_length
        self.mode = mode
        self._delta = torchaudio.transforms.ComputeDeltas(win_length=self.
            win_length, mode=self.mode)

    def forward(self, X):
        """
        Args:
             specgram (Tensor): Tensor of audio of dimension (..., freq, time).
        Returns:
            Tensor: specgram, deltas and double deltas of size (..., 3*freq, time).
        """
        delta = self._delta(X)
        double_delta = self._delta(delta)
        return torch.hstack((X, delta, double_delta))


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
