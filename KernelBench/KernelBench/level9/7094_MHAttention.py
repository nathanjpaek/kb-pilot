import math
import torch
from torch import nn
import torch.nn.functional as F


class MHAttention(nn.Module):

    def __init__(self, ninp, nhead, dropout):
        super(MHAttention, self).__init__()
        if ninp % nhead != 0:
            raise ValueError(
                'The hidden size is not a multiple of the number of attention heads'
                )
        self.nhead = nhead
        self.ninp = ninp
        self.fc_query = nn.Linear(ninp, ninp)
        self.fc_key = nn.Linear(ninp, ninp)
        self.fc_value = nn.Linear(ninp, ninp)
        self.dropout = nn.Dropout(dropout)

    def transpose_for_scores(self, x):
        """
        x has shape (*, L, C)
        return shape (*, nhead, L, C/nhead)
        """
        new_shape = x.shape[:-1] + (self.nhead, -1)
        x = x.view(*new_shape)
        return x.transpose(-3, -2)

    def forward_fn(self, x):
        """
        x has shape (*, L, C)
        return shape (*, L, C)
        """
        query = self.transpose_for_scores(self.fc_query(x))
        key = self.transpose_for_scores(self.fc_key(x))
        value = self.transpose_for_scores(self.fc_value(x))
        attention_scores = torch.matmul(query, key.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.ninp / self.nhead)
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        x = torch.matmul(attention_weights, value)
        x = x.transpose(-3, -2)
        x = x.reshape(*x.shape[:-2], -1)
        return x

    def forward(self, x):
        chunk_size = 100000 // x.shape[2]
        outputs = []
        for i in range(0, x.shape[1], chunk_size):
            ed = min(i + chunk_size, x.shape[1])
            partial = self.forward_fn(x[:, i:ed])
            outputs.append(partial)
        return torch.cat(outputs, dim=1)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'ninp': 4, 'nhead': 4, 'dropout': 0.5}]
