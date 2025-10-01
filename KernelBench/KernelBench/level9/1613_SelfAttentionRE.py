import torch
import torch.nn.functional as F
from torch import nn


class SelfAttentionRE(nn.Module):

    def __init__(self, emb_dim):
        super().__init__()
        self.query_mlp = nn.Linear(in_features=emb_dim, out_features=1)
        self.value_mlp = nn.Linear(in_features=emb_dim, out_features=emb_dim)

    def forward(self, x):
        B, N, N, emb_dim = x.shape
        value = self.value_mlp(x.view(-1, emb_dim)).view(B, N, N, emb_dim)
        query = self.query_mlp(x.view(-1, emb_dim)).view(B, N, N, 1)
        att_weights = F.softmax(query, dim=0)
        out = torch.sum(value * att_weights, dim=0)
        return out, att_weights.detach()


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'emb_dim': 4}]
