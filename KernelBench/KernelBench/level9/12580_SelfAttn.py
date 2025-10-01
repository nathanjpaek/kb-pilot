import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torch.nn.init
import torch as th


class SelfAttn(nn.Module):

    def __init__(self, hidden_size):
        super(SelfAttn, self).__init__()
        self.query = nn.Linear(hidden_size, 1)

    def forward(self, keys, values, attn_mask=None):
        """
        :param attn_inputs: batch_size x time_len x hidden_size
        :param attn_mask: batch_size x time_len
        :return: summary state
        """
        alpha = F.softmax(self.query(keys), dim=1)
        if attn_mask is not None:
            alpha = alpha * attn_mask.unsqueeze(2)
            alpha = alpha / th.sum(alpha, dim=1, keepdim=True)
        summary = th.sum(values * alpha, dim=1)
        return summary


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'hidden_size': 4}]
