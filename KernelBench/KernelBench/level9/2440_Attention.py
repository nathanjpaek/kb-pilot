import math
import torch
import torch.nn as nn


class Attention(nn.Module):
    """
        Using two types of attention mechanism: "Dot" and "Bahdanau"
    """

    def __init__(self, hidden_size, use_tanh=False, C=10, name='Bahdanau',
        use_cuda=True):
        super(Attention, self).__init__()
        self.use_tanh = use_tanh
        self.C = C
        self.name = name
        if name == 'Bahdanau':
            self.W_query = nn.Linear(hidden_size, hidden_size)
            self.W_ref = nn.Conv1d(hidden_size, hidden_size, 1, 1)
            V = torch.FloatTensor(hidden_size)
            if use_cuda:
                V = V
            self.V = nn.Parameter(V)
            self.V.data.uniform_(-(1.0 / math.sqrt(hidden_size)), 1.0 /
                math.sqrt(hidden_size))

    def forward(self, query, ref):
        """
        Args:
            query: [batch_size x hidden_size]
            ref:   [batch_size x seq_len x hidden_size]
        """
        batch_size = ref.size(0)
        seq_len = ref.size(1)
        if self.name == 'Bahdanau':
            ref = ref.permute(0, 2, 1)
            query = self.W_query(query).unsqueeze(2)
            ref = self.W_ref(ref)
            expanded_query = query.repeat(1, 1, seq_len)
            V = self.V.unsqueeze(0).unsqueeze(0).repeat(batch_size, 1, 1)
            logits = torch.bmm(V, torch.tanh(expanded_query + ref)).squeeze(1)
        elif self.name == 'Dot':
            query = query.unsqueeze(2)
            logits = torch.bmm(ref, query).squeeze(2)
            ref = ref.permute(0, 2, 1)
        else:
            raise NotImplementedError
        if self.use_tanh:
            logits = self.C * torch.tanh(logits)
        else:
            logits = logits
        return ref, logits


def get_inputs():
    return [torch.rand([4, 4]), torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'hidden_size': 4}]
