import torch
import torch.nn as nn
import torch.cuda


class FeedForward(nn.Module):

    def __init__(self, hidden_size, inner_size, dropout):
        super(FeedForward, self).__init__()
        self.linear_in = nn.Linear(hidden_size, inner_size, bias=False)
        self.linear_out = nn.Linear(inner_size, hidden_size, bias=False)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.linear_in.weight)
        nn.init.xavier_uniform_(self.linear_out.weight)

    def forward(self, x):
        y = self.linear_in(x)
        y = self.relu(y)
        y = self.dropout(y)
        y = self.linear_out(y)
        return y


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'hidden_size': 4, 'inner_size': 4, 'dropout': 0.5}]
