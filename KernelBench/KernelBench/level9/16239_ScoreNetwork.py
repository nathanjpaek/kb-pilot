from torch.nn import Module
import torch
from torch.nn import Tanh
from torch.nn import Linear


class ScoreNetwork(Module):
    """
    An optimized single hidden layer neural network for attention scores.
    The optimization idea behind this network is that projection of keys can
    performed only once without concatenation with query.

    It's allows to avoid unnecessary extra computations when attending every
    time-step over the same key-value pairs.
    """

    def __init__(self, query_dim, hidden_dim, non_linearity=Tanh()):
        super(ScoreNetwork, self).__init__()
        self.query_dim = query_dim
        self.hidden_dim = hidden_dim
        self.query_proj = Linear(query_dim, hidden_dim, bias=True)
        self.non_lin = non_linearity
        self.hidden_to_out_proj = Linear(hidden_dim, 1)

    def forward(self, query, key):
        """
        :param query: [batch_size, query_dim]
        :param key: [batch_size, seq_len, hidden_dim]
        :return: out: [batch_size, seq_len, 1]
        """
        assert key.size(2) == self.hidden_dim
        query = self.query_proj(query)
        hidden = self.non_lin(query.unsqueeze(1) + key)
        out = self.hidden_to_out_proj(hidden)
        return out


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'query_dim': 4, 'hidden_dim': 4}]
