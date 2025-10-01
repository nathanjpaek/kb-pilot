import torch
import torch.nn as nn


class RNNCell(nn.Module):

    def __init__(self, embed_dim, hidden_size, vocab_dim):
        super().__init__()
        self.hidden_size = hidden_size
        self.input2hidden = nn.Linear(embed_dim + hidden_size, hidden_size)

    def forward(self, inputs, hidden):
        combined = torch.cat((inputs, hidden), 2)
        hidden = torch.relu(self.input2hidden(combined))
        return hidden

    def init_hidden(self, batch_size):
        return torch.zeros(1, batch_size, self.hidden_size)


def get_inputs():
    return [torch.rand([4, 4, 4]), torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'embed_dim': 4, 'hidden_size': 4, 'vocab_dim': 4}]
