import torch
import torch.nn as nn


class MultAttention(nn.Module):
    """
    Multiplicative attention similar to Vaswani et al.
    """

    def __init__(self, key_dim: 'int', val_dim: 'int', out_dim: 'int'):
        super(MultAttention, self).__init__()
        self.key_encoder = nn.Linear(key_dim, out_dim)
        self.val_encoder = nn.Linear(val_dim, out_dim)
        self.query_encoder = nn.Linear(key_dim, out_dim)

    def forward(self, vals, keys_):
        """
        # Inputs:

        :param vals: Values of shape [batch x val_dim]
        :param keys: Keys of shape [batch x graphs x key_dim]
        """
        keys = self.key_encoder(keys_)
        queries = self.query_encoder(keys_)
        vals = self.val_encoder(vals)
        vals = vals.unsqueeze(1)
        weights = torch.matmul(keys, vals.transpose(1, 2))
        weights = torch.softmax(weights, 1)
        summed_queries = (queries * weights).sum(1)
        return summed_queries, weights


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'key_dim': 4, 'val_dim': 4, 'out_dim': 4}]
