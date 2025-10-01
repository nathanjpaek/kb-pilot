from _paritybench_helpers import _mock_config
import torch
from torch import nn


class Sampling(nn.Module):

    def __init__(self, args, seq_len):
        super(Sampling, self).__init__()
        self.conv = nn.Conv1d(seq_len, args.att_out_channel, kernel_size=1)

    def forward(self, x):
        """
        :param x: (batch, N=1, channel, wavelet_seq)
        :return:  (batch, N=1, att_out_channel, wavelet_seq[-1])
        """
        x = x.squeeze()
        conv_out = self.conv(x)
        return conv_out[..., -1]


def get_inputs():
    return [torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'args': _mock_config(att_out_channel=4), 'seq_len': 4}]
