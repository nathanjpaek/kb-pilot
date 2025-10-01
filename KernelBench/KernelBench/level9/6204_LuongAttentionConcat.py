import torch
import torch.nn as nn
import torch.nn.functional as F


class LuongAttentionConcat(nn.Module):

    def __init__(self, units, hidden_size):
        super().__init__()
        self.W = nn.Linear(2 * hidden_size, units)
        self.V = nn.Linear(units, 1)

    def forward(self, query, values):
        query = torch.squeeze(query, 0)
        query = torch.unsqueeze(query, 1)
        query = query.repeat(1, values.shape[1], 1)
        cat = torch.cat((values, query), dim=2)
        score = self.V(torch.tanh(self.W(cat)))
        attention_weights = F.softmax(score, dim=1)
        context_vector = attention_weights * values
        context_vector = context_vector.sum(1)
        return context_vector, attention_weights


def get_inputs():
    return [torch.rand([4, 4]), torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'units': 4, 'hidden_size': 4}]
