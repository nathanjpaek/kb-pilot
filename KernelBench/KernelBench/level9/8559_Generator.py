import torch
import torch.distributed
import torch
import torch.nn as nn


def gumbel_softmax(logits, tau=1.0, hard=False, log_mode=True, dim=-1):
    while True:
        gumbels = -torch.empty_like(logits).exponential_().log()
        gumbels = (logits + gumbels) / tau
        if log_mode:
            y_soft = gumbels.log_softmax(dim)
        else:
            y_soft = gumbels.softmax(dim)
        if torch.sum(torch.isnan(y_soft)).item() < 0.01:
            break
    if hard:
        index = y_soft.max(dim, keepdim=True)[1]
        y_hard = torch.zeros_like(logits).scatter_(dim, index, 1.0)
        ret = y_hard - y_soft.detach() + y_soft
    else:
        ret = y_soft
    return ret


class Generator(nn.Module):

    def __init__(self, vocab_size, dec_hidden_size, pad_idx):
        super(Generator, self).__init__()
        self.linear = nn.Linear(dec_hidden_size, vocab_size)
        self.softmax = nn.LogSoftmax(dim=-1)
        self.pad_idx = pad_idx

    def forward(self, x, use_gumbel_softmax=False):
        output = self.linear(x)
        output[:, self.pad_idx] = -float('inf')
        if use_gumbel_softmax:
            output = gumbel_softmax(output, log_mode=True, dim=-1)
        else:
            output = self.softmax(output)
        return output


def get_inputs():
    return [torch.rand([4, 5, 4, 4])]


def get_init_inputs():
    return [[], {'vocab_size': 4, 'dec_hidden_size': 4, 'pad_idx': 4}]
