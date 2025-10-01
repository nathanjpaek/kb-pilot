import torch
import torch.nn as nn
from torch.autograd import *


class MutualBiAffineAttention(nn.Module):
    """
    Mutual BiAffine Attention between 2 kinds of features.
    """

    def __init__(self, hidden_size):
        super(MutualBiAffineAttention, self).__init__()
        self.linear1 = nn.Linear(2 * hidden_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, 1)

    def forward(self, in1, att_feats, att_mask):
        att = self.linear2(torch.tanh(self.linear1(torch.cat([in1,
            att_feats], -1))))
        att_mask = att_mask.unsqueeze(-1)
        att = torch.softmax(att, -1) * att_mask
        att = att / (att.sum(-1, keepdim=True) + 1e-20)
        att_res = att * att_feats
        return att_res


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand(
        [4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'hidden_size': 4}]
