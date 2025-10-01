import torch
import torch.nn.functional as F
import torch.nn as nn


class StyleEmbedAttention(nn.Module):
    """ StyleEmbedAttention """

    def __init__(self, query_dim, key_dim, num_units, num_heads):
        super(StyleEmbedAttention, self).__init__()
        self.num_units = num_units
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.W_query = nn.Linear(in_features=query_dim, out_features=
            num_units, bias=False)
        self.W_key = nn.Linear(in_features=key_dim, out_features=num_units,
            bias=False)
        self.W_value = nn.Linear(in_features=key_dim, out_features=
            num_units, bias=False)

    def forward(self, query, key_soft):
        """
        input:
            query --- [N, T_q, query_dim]
            key_soft --- [N, T_k, key_dim]
        output:
            out --- [N, T_q, num_units]
        """
        values = self.W_value(key_soft)
        split_size = self.num_units // self.num_heads
        values = torch.stack(torch.split(values, split_size, dim=2), dim=0)
        out_soft = scores_soft = None
        querys = self.W_query(query)
        keys = self.W_key(key_soft)
        querys = torch.stack(torch.split(querys, split_size, dim=2), dim=0)
        keys = torch.stack(torch.split(keys, split_size, dim=2), dim=0)
        scores_soft = torch.matmul(querys, keys.transpose(2, 3))
        scores_soft = scores_soft / self.key_dim ** 0.5
        scores_soft = F.softmax(scores_soft, dim=3)
        out_soft = torch.matmul(scores_soft, values)
        out_soft = torch.cat(torch.split(out_soft, 1, dim=0), dim=3).squeeze(0)
        return out_soft


def get_inputs():
    return [torch.rand([4, 4, 4]), torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'query_dim': 4, 'key_dim': 4, 'num_units': 4, 'num_heads': 4}]
