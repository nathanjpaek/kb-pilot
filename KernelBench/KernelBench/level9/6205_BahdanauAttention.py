import torch
import torch.nn as nn
import torch.nn.functional as F


class BahdanauAttention(nn.Module):

    def __init__(self, units, hidden_size):
        super().__init__()
        self.W1 = nn.Linear(hidden_size, units)
        self.W2 = nn.Linear(hidden_size, units)
        self.V = nn.Linear(units, 1)

    def forward(self, query, values):
        query = torch.squeeze(query, 0)
        hidden_with_time_axis = torch.unsqueeze(query, 1)
        sum_1 = self.W1(values) + self.W2(hidden_with_time_axis)
        score = self.V(torch.tanh(sum_1))
        attention_weights = F.softmax(score, dim=1)
        context_vector = attention_weights * values
        context_vector = context_vector.sum(1)
        return context_vector, attention_weights


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'units': 4, 'hidden_size': 4}]
