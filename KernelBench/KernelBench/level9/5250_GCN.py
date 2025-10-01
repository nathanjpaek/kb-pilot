import torch
import torch.nn as nn
import torch.utils.data


class GCN(nn.Module):
    """
    Graph Convolutional Network based on https://arxiv.org/abs/1609.02907

    """

    def __init__(self, feat_dim, hidden_dim1, hidden_dim2, dropout,
        is_sparse=False):
        """Dense version of GAT."""
        super(GCN, self).__init__()
        self.W1 = nn.Parameter(torch.FloatTensor(feat_dim, hidden_dim1))
        self.W2 = nn.Parameter(torch.FloatTensor(hidden_dim1, hidden_dim2))
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)
        nn.init.xavier_uniform_(self.W1.data)
        nn.init.xavier_uniform_(self.W2.data)
        self.is_sparse = is_sparse

    def forward(self, x, adj):
        support = torch.mm(x, self.W1)
        embeddings = torch.sparse.mm(adj, support
            ) if self.is_sparse else torch.mm(adj, support)
        embeddings = self.relu(embeddings)
        embeddings = self.dropout(embeddings)
        support = torch.mm(embeddings, self.W2)
        embeddings = torch.sparse.mm(adj, support
            ) if self.is_sparse else torch.mm(adj, support)
        embeddings = self.relu(embeddings)
        return embeddings


def get_inputs():
    return [torch.rand([4, 4]), torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'feat_dim': 4, 'hidden_dim1': 4, 'hidden_dim2': 4,
        'dropout': 0.5}]
