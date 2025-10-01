import torch
import torch.nn as nn


class BahdanauAttn(nn.Module):
    """Bahdabau attention mechanism"""

    def __init__(self, size):
        super(BahdanauAttn, self).__init__()
        self.query_layer = nn.Linear(size, size, bias=False)
        self.tanh = nn.Tanh()
        self.v = nn.Linear(size, 1, bias=False)

    def forward(self, query, memory):
        """
        Args:
            query: (batch_size, 1, size) or (batch_size, size)
            memory: (batch_size, timesteps, size)
        """
        if query.dim() == 2:
            query = query.unsqueeze(1)
        Q = self.query_layer(query)
        K = memory
        alignment = self.v(self.tanh(Q + K))
        alignment = alignment.squeeze(-1)
        return alignment


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'size': 4}]
