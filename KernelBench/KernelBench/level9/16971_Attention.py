import torch
import torch.nn as nn


def weight_init(m):
    if isinstance(m, nn.Linear):
        size = m.weight.size()
        size[0]
        size[1]
        variance = 0.001
        m.weight.data.normal_(0.0, variance)
        try:
            m.bias.data.normal_(0.0, 0.0001)
        except:
            pass


class Attention(nn.Module):

    def __init__(self, hidden_size, query_size, use_softmax=False):
        super(Attention, self).__init__()
        self.use_softmax = use_softmax
        self.W_query = nn.Linear(query_size, hidden_size, bias=True)
        self.W_ref = nn.Linear(hidden_size, hidden_size, bias=False)
        V = torch.normal(torch.zeros(hidden_size), 0.0001)
        self.V = nn.Parameter(V)
        weight_init(V)
        weight_init(self.W_query)
        weight_init(self.W_ref)

    def forward(self, query, ref):
        """
        Args:
            query: [hidden_size]
            ref:   [seq_len x hidden_size]
        """
        ref.size(0)
        query = self.W_query(query)
        _ref = self.W_ref(ref)
        m = torch.tanh(query + _ref)
        logits = torch.matmul(m, self.V)
        if self.use_softmax:
            logits = torch.softmax(logits, dim=0)
        else:
            logits = logits
        return logits


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'hidden_size': 4, 'query_size': 4}]
