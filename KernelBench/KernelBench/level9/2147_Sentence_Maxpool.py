import torch
import torch as th
import torch.nn.functional as F
import torch.nn as nn


class Sentence_Maxpool(nn.Module):

    def __init__(self, word_dimension, output_dim, relu=True):
        super(Sentence_Maxpool, self).__init__()
        self.fc = nn.Linear(word_dimension, output_dim)
        self.out_dim = output_dim
        self.relu = relu

    def forward(self, x):
        x = self.fc(x)
        if self.relu:
            x = F.relu(x)
        return th.max(x, dim=1)[0]


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'word_dimension': 4, 'output_dim': 4}]
