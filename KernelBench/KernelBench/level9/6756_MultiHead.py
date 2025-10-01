import torch
import torch.nn as nn


class Attention(nn.Module):

    def __init__(self):
        super().__init__()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, Q, K, V, mask=None, dk=64):
        w = torch.bmm(Q, K.transpose(1, 2))
        if mask is not None:
            assert w.size() == mask.size()
            w.masked_fill_(mask, -float('inf'))
        w = self.softmax(w / dk ** 0.5)
        c = torch.bmm(w, V)
        return c


class MultiHead(nn.Module):

    def __init__(self, hidden_size, n_splits):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_splits = n_splits
        self.Q_linear = nn.Linear(hidden_size, hidden_size, bias=False)
        self.K_linear = nn.Linear(hidden_size, hidden_size, bias=False)
        self.V_linear = nn.Linear(hidden_size, hidden_size, bias=False)
        self.linear = nn.Linear(hidden_size, hidden_size, bias=False)
        self.attn = Attention()

    def forward(self, Q, K, V, mask=None):
        QWs = self.Q_linear(Q).split(self.hidden_size // self.n_splits, dim=-1)
        KWs = self.K_linear(K).split(self.hidden_size // self.n_splits, dim=-1)
        VWs = self.V_linear(V).split(self.hidden_size // self.n_splits, dim=-1)
        QWs = torch.cat(QWs, dim=0)
        KWs = torch.cat(KWs, dim=0)
        VWs = torch.cat(VWs, dim=0)
        if mask is not None:
            mask = torch.cat([mask for _ in range(self.n_splits)], dim=0)
        c = self.attn(QWs, KWs, VWs, mask=mask, dk=self.hidden_size // self
            .n_splits)
        c = c.split(Q.size(0), dim=0)
        c = self.linear(torch.cat(c, dim=-1))
        return c


def get_inputs():
    return [torch.rand([4, 4, 4]), torch.rand([4, 4, 4]), torch.rand([4, 4, 4])
        ]


def get_init_inputs():
    return [[], {'hidden_size': 4, 'n_splits': 4}]
