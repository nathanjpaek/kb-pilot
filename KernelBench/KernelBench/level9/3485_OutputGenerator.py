import torch
import torch.nn as nn


class OutputGenerator(nn.Module):

    def __init__(self, model_dim, tgt_vocab_size):
        super().__init__()
        self.tgt_vocab_size = tgt_vocab_size
        self.linear = nn.Linear(model_dim, tgt_vocab_size, bias=False)
        self.log_softmax = nn.LogSoftmax(dim=-1)

    def forward(self, tgt):
        tgt_log_probs = self.log_softmax(self.linear(tgt))
        return tgt_log_probs.reshape(-1, self.tgt_vocab_size)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'model_dim': 4, 'tgt_vocab_size': 4}]
