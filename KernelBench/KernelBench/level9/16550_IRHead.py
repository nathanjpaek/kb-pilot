import torch
import torch.nn as nn
from queue import *
from math import *


class IRHead(nn.Module):

    def __init__(self, hidden_size, dropout=0.5):
        super(IRHead, self).__init__()
        self.M = nn.Parameter(torch.randn(hidden_size, hidden_size))
        self.hidden_layer = nn.Linear(hidden_size * 2 + 1, hidden_size)
        self.opt_layer = nn.Linear(hidden_size, 2)
        self.hidden_drop = nn.Dropout(p=dropout)

    def forward(self, src_embed, tgt_embed):
        """
        src_embed: [batch, hidden]
        tgt_embed: [batch, hidden]

        return the score: [batch, 2]
        """
        src_hidden = src_embed.unsqueeze(1)
        tgt_hidden = tgt_embed.unsqueeze(2)
        score = torch.bmm(torch.matmul(src_hidden, self.M), tgt_hidden
            ).squeeze(2)
        src_hidden = src_hidden.squeeze(1)
        tgt_hidden = tgt_hidden.squeeze(2)
        inpt = torch.cat([src_hidden, score, tgt_hidden], 1)
        inpt = self.hidden_drop(torch.tanh(self.hidden_layer(inpt)))
        score = self.opt_layer(inpt)
        return score


def get_inputs():
    return [torch.rand([4, 4]), torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'hidden_size': 4}]
