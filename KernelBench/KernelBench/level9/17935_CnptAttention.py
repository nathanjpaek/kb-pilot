import torch
from torch import nn


class CnptAttention(nn.Module):

    def __init__(self, in_dim, out_dim):
        super(CnptAttention, self).__init__()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, query, key):
        """
        query: sent_emb (1, D)
        key: [(k, D), (k,D)]
        value: (k, kg_dim, kg_dim)
        """
        num = key[0].shape[0]
        query = query.view(1, query.shape[-1]).expand(num, -1)
        h_score = -torch.pow(query - key[0], 2).sum(-1).unsqueeze(-1)
        t_score = -torch.pow(query - key[1], 2).sum(-1).unsqueeze(-1)
        h_t_score = torch.cat((h_score, t_score), -1)
        score, _ = torch.max(h_t_score, -1)
        score = self.softmax(score)
        return score.squeeze()


def get_inputs():
    return [torch.rand([1, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_dim': 4, 'out_dim': 4}]
