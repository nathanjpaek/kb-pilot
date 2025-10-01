import torch
import torch.nn as nn
from collections import OrderedDict


class Conv1D(nn.Module):

    def __init__(self, embedding_dim, hidden_dim):
        super(Conv1D, self).__init__()
        self.convs = nn.Sequential(OrderedDict([('conv1', nn.Conv1d(
            embedding_dim, hidden_dim, kernel_size=3, stride=1, padding=2)),
            ('relu1', nn.ReLU()), ('conv2', nn.Conv1d(hidden_dim,
            hidden_dim, 3, 1, 2)), ('relu2', nn.ReLU()), ('conv3', nn.
            Conv1d(hidden_dim, hidden_dim, 3, 1, 2)), ('tanh', nn.Tanh())]))

    def forward(self, embedding):
        return self.convs(embedding.transpose(-2, -1)).transpose(-2, -1)


def get_inputs():
    return [torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'embedding_dim': 4, 'hidden_dim': 4}]
