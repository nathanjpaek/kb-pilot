import torch
import torch as th
import torch.nn as nn


class ScalarFilter(nn.Module):

    def __init__(self):
        super(ScalarFilter, self).__init__()

    def forward(self, p_x, g_x):
        """
        input should be scalar: bsz x l1, bsz x l2
        return bsz x l2
        """
        matrix = g_x.unsqueeze(2) - p_x.unsqueeze(1)
        return th.max(matrix == 0, dim=2)[0]


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
