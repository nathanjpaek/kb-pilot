import torch
import torch.nn as nn
from sklearn.metrics import *


class SelfAttentive(nn.Module):

    def __init__(self, hidden_size, att_hops=1, att_unit=200, dropout=0.2):
        super(SelfAttentive, self).__init__()
        self.drop = nn.Dropout(dropout)
        self.ws1 = nn.Linear(hidden_size, att_unit, bias=False)
        self.ws2 = nn.Linear(att_unit, att_hops, bias=False)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax()
        self.attention_hops = att_hops

    def forward(self, rnn_out, mask=None):
        outp = rnn_out
        size = outp.size()
        compressed_embeddings = outp.reshape(-1, size[2])
        hbar = self.tanh(self.ws1(self.drop(compressed_embeddings)))
        alphas = self.ws2(hbar).view(size[0], size[1], -1)
        alphas = torch.transpose(alphas, 1, 2).contiguous()
        if mask is not None:
            mask = mask.squeeze(2)
            concatenated_mask = [mask for i in range(self.attention_hops)]
            concatenated_mask = torch.cat(concatenated_mask, 1)
            penalized_alphas = alphas + concatenated_mask
        else:
            penalized_alphas = alphas
        alphas = self.softmax(penalized_alphas.view(-1, size[1]))
        alphas = alphas.view(size[0], self.attention_hops, size[1])
        return torch.bmm(alphas, outp), alphas


def get_inputs():
    return [torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'hidden_size': 4}]
