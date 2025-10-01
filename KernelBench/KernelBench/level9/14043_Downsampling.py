from torch.nn import Module
import torch
from torch.nn import Sequential
from torch.nn import Linear


class FullyConnected(torch.nn.Module):

    def __init__(self, in_features, out_features, bias=True, activation=None):
        super().__init__()
        self.linear = torch.nn.Linear(in_features, out_features, bias=bias)
        if activation is None:
            self.activation = torch.nn.Identity()
        elif activation == 'relu':
            self.activation = torch.nn.ReLU()
        elif activation == 'elu':
            self.activation = torch.nn.ELU(alpha=1.0)
        elif activation == 'lrelu':
            self.activation = torch.nn.LeakyReLU(0.1)
        else:
            raise ValueError()

    def forward(self, x):
        return self.activation(self.linear(x))


class GPool(Module):

    def __init__(self, n, dim, use_mlp=False, mlp_activation='relu'):
        super().__init__()
        self.use_mlp = use_mlp
        if use_mlp:
            self.pre = Sequential(FullyConnected(dim, dim // 2, bias=True,
                activation=mlp_activation), FullyConnected(dim // 2, dim //
                4, bias=True, activation=mlp_activation))
            self.p = Linear(dim // 4, 1, bias=True)
        else:
            self.p = Linear(dim, 1, bias=True)
        self.n = n

    def forward(self, pos, x):
        batchsize = x.size(0)
        if self.n < 1:
            k = int(x.size(1) * self.n)
        else:
            k = self.n
        if self.use_mlp:
            y = self.pre(x)
        else:
            y = x
        y = (self.p(y) / torch.norm(self.p.weight, p='fro')).squeeze(-1)
        top_idx = torch.argsort(y, dim=1, descending=True)[:, 0:k]
        y = torch.gather(y, dim=1, index=top_idx)
        y = torch.sigmoid(y)
        pos = torch.gather(pos, dim=1, index=top_idx.unsqueeze(-1).expand(
            batchsize, k, 3))
        x = torch.gather(x, dim=1, index=top_idx.unsqueeze(-1).expand(
            batchsize, k, x.size(-1)))
        x = x * y.unsqueeze(-1).expand_as(x)
        return top_idx, pos, x


class Downsampling(Module):

    def __init__(self, feature_dim, ratio=0.5):
        super().__init__()
        self.pool = GPool(ratio, dim=feature_dim)

    def forward(self, pos, x):
        """
        :param  pos:    (B, N, 3)
        :param  x:      (B, N, d)
        :return (B, rN, d)
        """
        idx, pos, x = self.pool(pos, x)
        return idx, pos, x


def get_inputs():
    return [torch.rand([4, 4, 4]), torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'feature_dim': 4}]
