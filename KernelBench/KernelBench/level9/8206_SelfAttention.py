import torch
from torch import nn
import torch.nn.modules.loss
from scipy.sparse import *


class SelfAttention(nn.Module):

    def __init__(self, input_size, hidden_size):
        super(SelfAttention, self).__init__()
        self.W1 = torch.Tensor(input_size, hidden_size)
        self.W1 = nn.Parameter(nn.init.xavier_uniform_(self.W1))
        self.W2 = torch.Tensor(hidden_size, 1)
        self.W2 = nn.Parameter(nn.init.xavier_uniform_(self.W2))

    def forward(self, x, attention_mask=None):
        attention = torch.mm(torch.tanh(torch.mm(x.view(-1, x.size(-1)),
            self.W1)), self.W2).view(x.size(0), -1)
        if attention_mask is not None:
            attention = attention.masked_fill_(1 - attention_mask.byte(), -INF)
        probs = torch.softmax(attention, dim=-1).unsqueeze(1)
        weighted_x = torch.bmm(probs, x).squeeze(1)
        return weighted_x


def get_inputs():
    return [torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'input_size': 4, 'hidden_size': 4}]
