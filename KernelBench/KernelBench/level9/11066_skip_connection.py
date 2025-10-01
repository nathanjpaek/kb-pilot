import torch
import torch.nn as nn


class skip_connection(nn.Module):

    def __init__(self, inchannel, outchannel, keep_dim=True):
        super(skip_connection, self).__init__()
        if inchannel != outchannel:
            self.conv1d = nn.Conv1d(inchannel, outchannel, 1)

    def forward(self, before, after):
        """
        :param before: the tensor before passing convolution blocks
        :param after: the tensor of output from convolution blocks
        :return: the sum of inputs
        """
        if before.shape[2] != after.shape[2]:
            before = nn.functional.max_pool1d(before, 2, 2)
        if before.shape[1] != after.shape[1]:
            before = self.conv1d(before)
        return before + after


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'inchannel': 4, 'outchannel': 4}]
