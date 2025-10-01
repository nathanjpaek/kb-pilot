import torch
import torch.nn as nn
import torch.cuda


class Generator(nn.Module):

    def __init__(self, hidden_size: 'int', tgt_vocab_size: 'int'):
        self.vocab_size = tgt_vocab_size
        super(Generator, self).__init__()
        self.linear_hidden = nn.Linear(hidden_size, tgt_vocab_size)
        self.lsm = nn.LogSoftmax(dim=-1)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.linear_hidden.weight)
        nn.init.constant_(self.linear_hidden.bias, 0.0)

    def forward(self, dec_out):
        score = self.linear_hidden(dec_out)
        lsm_score = self.lsm(score)
        return lsm_score.view(-1, self.vocab_size)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'hidden_size': 4, 'tgt_vocab_size': 4}]
