import torch
from torch import nn


class BehaviorAggregator(nn.Module):

    def __init__(self, embedding_dim, gamma=0.5, aggregator='mean',
        dropout_rate=0.0):
        super(BehaviorAggregator, self).__init__()
        self.aggregator = aggregator
        self.gamma = gamma
        self.W_v = nn.Linear(embedding_dim, embedding_dim, bias=False)
        if self.aggregator in ['cross_attention', 'self_attention']:
            self.W_k = nn.Sequential(nn.Linear(embedding_dim, embedding_dim
                ), nn.Tanh())
            self.dropout = nn.Dropout(dropout_rate
                ) if dropout_rate > 0 else None
            if self.aggregator == 'self_attention':
                self.W_q = nn.Parameter(torch.Tensor(embedding_dim, 1))
                nn.init.xavier_normal_(self.W_q)

    def forward(self, id_emb, sequence_emb):
        out = id_emb
        if self.aggregator == 'mean':
            out = self.average_pooling(sequence_emb)
        elif self.aggregator == 'cross_attention':
            out = self.cross_attention(id_emb, sequence_emb)
        elif self.aggregator == 'self_attention':
            out = self.self_attention(sequence_emb)
        return self.gamma * id_emb + (1 - self.gamma) * out

    def cross_attention(self, id_emb, sequence_emb):
        key = self.W_k(sequence_emb)
        mask = sequence_emb.sum(dim=-1) == 0
        attention = torch.bmm(key, id_emb.unsqueeze(-1)).squeeze(-1)
        attention = self.masked_softmax(attention, mask)
        if self.dropout is not None:
            attention = self.dropout(attention)
        output = torch.bmm(attention.unsqueeze(1), sequence_emb).squeeze(1)
        return self.W_v(output)

    def self_attention(self, sequence_emb):
        key = self.W_k(sequence_emb)
        mask = sequence_emb.sum(dim=-1) == 0
        attention = torch.matmul(key, self.W_q).squeeze(-1)
        attention = self.masked_softmax(attention, mask)
        if self.dropout is not None:
            attention = self.dropout(attention)
        output = torch.bmm(attention.unsqueeze(1), sequence_emb).squeeze(1)
        return self.W_v(output)

    def average_pooling(self, sequence_emb):
        mask = sequence_emb.sum(dim=-1) != 0
        mean = sequence_emb.sum(dim=1) / (mask.float().sum(dim=-1, keepdim=
            True) + 1e-12)
        return self.W_v(mean)

    def masked_softmax(self, X, mask):
        X = X.masked_fill_(mask, 0)
        e_X = torch.exp(X)
        return e_X / (e_X.sum(dim=1, keepdim=True) + 1e-12)


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'embedding_dim': 4}]
