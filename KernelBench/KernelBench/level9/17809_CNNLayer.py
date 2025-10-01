import torch
import torch.nn as nn


class CNNLayer(nn.Module):
    """Conv1d layer.
    nn.Conv1d layer require the input shape is (batch_size, in_channels, length),
    however, our input shape is (batch_size, length, in_channels), so we need to
    transpose our input data into (B, C, L_in) and send it to conv layer, and
    then transpose the conv output into (B, L_out, C).
    """

    def __init__(self, in_dim, out_dim, win=3, pad=1):
        super().__init__()
        self.conv = nn.Conv1d(in_channels=in_dim, out_channels=out_dim,
            kernel_size=win, padding=pad)

    def forward(self, x):
        """
        Args:
            x: shape=(batch_size, max_seq_len, input_dim)
        """
        x = x.permute(0, 2, 1)
        out = self.conv(x).permute(0, 2, 1)
        return out


def get_inputs():
    return [torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'in_dim': 4, 'out_dim': 4}]
