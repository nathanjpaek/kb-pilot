import torch
from torch import nn


class BahdanauAttention(nn.Module):

    def __init__(self, dim):
        super(BahdanauAttention, self).__init__()
        self.query_layer = nn.Linear(dim, dim, bias=False)
        self.tanh = nn.Tanh()
        self.v = nn.Linear(dim, 1, bias=False)

    def forward(self, query, processed_memory):
        """
        Args:
            query: (batch, 1, dim) or (batch, dim)
            processed_memory: (batch, max_time, dim)
        """
        if query.dim() == 2:
            query = query.unsqueeze(1)
        processed_query = self.query_layer(query)
        alignment = self.v(self.tanh(processed_query + processed_memory))
        return alignment.squeeze(-1)


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'dim': 4}]
