import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class BaseSelfAttention(nn.Module):

    def __init__(self):
        super(BaseSelfAttention, self).__init__()

    def init_linear(self, input_linear):
        """Initialize linear transformation"""
        bias = np.sqrt(6.0 / (input_linear.weight.size(0) + input_linear.
            weight.size(1)))
        nn.init.uniform_(input_linear.weight, -bias, bias)
        if input_linear.bias is not None:
            input_linear.bias.data.zero_()

    def initialize_layers(self):
        raise NotImplementedError

    def forward(self, X):
        raise NotImplementedError

    def score(self, a, b):
        raise NotImplementedError


class SelfAttention(BaseSelfAttention):

    def __init__(self, hidden_dim, scoring='general'):
        super(SelfAttention, self).__init__()
        self.scoring = scoring
        self.hidden_dim = hidden_dim
        self.initialize_layers()

    def initialize_layers(self):
        if self.scoring == 'general':
            self.W = nn.Linear(self.hidden_dim, self.hidden_dim)
            self.init_linear(self.W)
        elif self.scoring == 'concat':
            self.W = nn.Linear(2 * self.hidden_dim, self.hidden_dim)
            self.v = nn.Linear(self.hidden_dim, 1)
            self.init_linear(self.W)
            self.init_linear(self.v)
        elif self.scoring == 'dot':
            pass
        else:
            raise RuntimeError('Unrecognized attention scoring method: %s' %
                self.scoring)

    def forward(self, hidden_outputs):
        scores = self.score(hidden_outputs)
        context = scores.bmm(hidden_outputs)
        return context

    def score(self, hidden_outputs):
        if self.scoring == 'dot':
            H = hidden_outputs.transpose(1, 2)
            attention_energies = hidden_outputs.bmm(H)
            scores = F.softmax(attention_energies, dim=2)
            return scores
        elif self.scoring == 'general':
            H = self.W(hidden_outputs)
            H = H.transpose(1, 2)
            attention_energies = hidden_outputs.bmm(H)
            scores = F.softmax(attention_energies, dim=2)
            return scores
        elif self.scoring == 'concat':
            H = hidden_outputs.transpose(1, 2)
            scores = []
            batch_size, doc_maxlen, hidden_dim = hidden_outputs.shape
            for doc_idx in range(H.shape[-1]):
                h_t = hidden_outputs[:, doc_idx, :]
                h_t = h_t.unsqueeze(1)
                h_t = h_t.repeat(1, doc_maxlen, 1)
                H_t = torch.cat((h_t, hidden_outputs), dim=2)
                H_t = self.W(H_t)
                H_t = torch.nn.functional.tanh(H_t)
                H_t = self.v(H_t)
                H_t = H_t.view(batch_size, doc_maxlen)
                scores.append(H_t)
            scores = torch.stack(scores)
            scores = scores.transpose(0, 1)
            scores = scores / torch.sqrt(torch.Tensor([hidden_dim]))
            scores = F.softmax(scores, dim=2)
            return scores
        else:
            raise RuntimeError('Unrecognized scoring method: %s' % self.scoring
                )


def get_inputs():
    return [torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'hidden_dim': 4}]
