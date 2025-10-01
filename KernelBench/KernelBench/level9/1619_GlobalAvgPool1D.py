import torch
import torch.nn.functional as functional


class GlobalAvgPool1D(torch.nn.Module):

    def __init__(self):
        super(GlobalAvgPool1D, self).__init__()

    def forward(self, x):
        """
        x shape: (batch_size, channel, seq_len)
        return shape: (batch_size, channel, 1)
        """
        return functional.avg_pool1d(x, kernel_size=x.shape[2])


def get_inputs():
    return [torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {}]
