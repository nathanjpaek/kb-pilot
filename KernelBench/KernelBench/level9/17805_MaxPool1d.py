import torch
import torch.nn as nn


class MaxPool1d(nn.Module):

    def __init__(self, win=2, stride=None, pad=0):
        super().__init__()
        self.pooling = nn.MaxPool1d(kernel_size=win, stride=stride, padding=pad
            )

    def forward(self, x):
        """
        Args:
            x: shape=(batch_size, max_seq_len, dim)
        """
        x = x.permute(0, 2, 1)
        x = self.pooling(x).permute(0, 2, 1)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {}]
