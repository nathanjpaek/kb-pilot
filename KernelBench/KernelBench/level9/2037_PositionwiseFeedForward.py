import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torch.multiprocessing


class PositionwiseFeedForward(nn.Module):
    """ A two-feed-forward-layer module """

    def __init__(self, d_in, d_hid, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Conv1d(d_in, d_hid, 1)
        self.w_2 = nn.Conv1d(d_hid, d_in, 1)
        self.layer_norm = nn.LayerNorm(d_hid)

    def forward(self, x):
        output = self.w_1(x.transpose(1, 2)).transpose(1, 2)
        output = F.relu(self.layer_norm(output))
        output = self.w_2(output.transpose(1, 2)).transpose(1, 2)
        return output


def get_inputs():
    return [torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'d_in': 4, 'd_hid': 4}]
