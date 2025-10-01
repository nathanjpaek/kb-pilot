import torch
import torch.nn.functional as functional


class GlobalMaxPool1D(torch.nn.Module):

    def __init__(self):
        super(GlobalMaxPool1D, self).__init__()

    def forward(self, x):
        """
        x shape: (batch_size, channel, seq_len)
        return shape: (batch_size, channel, 1)
        """
        return functional.max_pool1d(x, kernel_size=x.shape[2])


def get_inputs():
    return [torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {}]
