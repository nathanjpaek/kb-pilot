import torch
import torch.nn.functional as F
import torch.nn as nn


class SparseDropout(nn.Module):

    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, x):
        if not self.training:
            return x
        x_coal = x.coalesce()
        drop_val = F.dropout(x_coal._values(), self.p, self.training)
        return torch.sparse.FloatTensor(x_coal._indices(), drop_val, x.shape)


class MixedDropout(nn.Module):

    def __init__(self, p=0.5):
        super().__init__()
        self.dense_dropout = nn.Dropout(p)
        self.sparse_dropout = SparseDropout(p)

    def forward(self, x):
        if x.is_sparse:
            return self.sparse_dropout(x)
        else:
            return self.dense_dropout(x)


class APPNProp(nn.Module):

    def __init__(self, alpha: 'float'=0.1, K: 'int'=10, dropout: 'float'=0.0):
        super().__init__()
        self.alpha = alpha
        self.K = K
        if not dropout:
            self.dropout = lambda x: x
        else:
            self.dropout = MixedDropout(dropout)

    def forward(self, x, adj):
        h = x
        for _ in range(self.K):
            A_drop = self.dropout(adj)
            h = (1 - self.alpha) * A_drop.mm(h) + self.alpha * x
        return h

    def __repr__(self):
        return (
            f'{self.__class__.__name__}(alpha={self.alpha}, K={self.K}, dropout={self.dropout})'
            )


def get_inputs():
    return [torch.rand([4, 4]), torch.rand([4, 4])]


def get_init_inputs():
    return [[], {}]
