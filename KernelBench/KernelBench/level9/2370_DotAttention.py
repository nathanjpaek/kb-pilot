import torch
from torch import nn
import torch.optim


class AttentionMechanism(nn.Module):

    def __init__(self):
        super(AttentionMechanism, self).__init__()

    def forward(self, *input):
        raise NotImplementedError('Implement this.')


class DotAttention(AttentionMechanism):

    def __init__(self):
        super(DotAttention, self).__init__()

    def forward(self, q, k):
        return q @ k.transpose(1, 2)


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
