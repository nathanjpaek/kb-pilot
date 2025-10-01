import torch
import numpy as np
import torch.utils.data
from torch import nn
import torch
from torch.nn import functional as F


class QueryAttentionAggregator(nn.Module):

    def __init__(self, input_dim):
        super(QueryAttentionAggregator, self).__init__()
        self.query = nn.Linear(input_dim, input_dim)
        self.vector = nn.Parameter(torch.zeros(input_dim))
        self.input_dim = input_dim

    def forward(self, x, mask=None, agg_dim=1):
        keys = torch.sigmoid(self.query(x))
        dot_products = keys.matmul(self.vector) / np.sqrt(self.input_dim)
        dot_products = dot_products.unsqueeze(-1)
        if mask is not None:
            dot_products = dot_products - 1000 * mask
        attention_weights = F.softmax(dot_products, dim=-2).expand(-1, -1,
            self.input_dim)
        return (attention_weights * x).sum(-2)


def get_inputs():
    return [torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'input_dim': 4}]
