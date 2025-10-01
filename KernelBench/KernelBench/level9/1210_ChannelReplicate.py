import torch
import torch.nn as nn


class ChannelReplicate(nn.Module):

    def __init__(self, factor=3):
        super(ChannelReplicate, self).__init__()
        self.factor = factor

    def forward(self, input):
        template = input
        for i in range(0, self.factor - 1):
            input = torch.cat((template, input), 1)
        return input


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
