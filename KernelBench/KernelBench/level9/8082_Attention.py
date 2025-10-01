import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def init_linear_wt(linear):
    nn.init.xavier_uniform_(linear.weight)
    if linear.bias is not None:
        n = linear.bias.size(0)
        start, end = n // 4, n // 2
        linear.bias.data.fill_(0.0)
        linear.bias.data[start:end].fill_(1.0)


class Attention(nn.Module):

    def __init__(self, H, method='general'):
        super(Attention, self).__init__()
        self.method = method
        if self.method == 'general':
            self.W = nn.Linear(H, H)
            init_linear_wt(self.W)
        elif self.method == 'concat':
            self.W = nn.Linear(H * 2, H)
            self.v = nn.Parameter(torch.FloatTensor(1, H))
            init_linear_wt(self.W)
            stdv = 1.0 / math.sqrt(self.v.size(0))
            self.v.data.normal_(mean=0, std=stdv)
        self.W_c = nn.Linear(H * 2, H)
        init_linear_wt(self.W_c)

    def forward(self, K, V, Q):
        e = F.softmax(self.score(K, Q), dim=2)
        c = torch.bmm(e, V)
        h = torch.tanh(self.W_c(torch.cat((c, Q), dim=2)))
        return h, e

    def score(self, K, Q):
        if self.method == 'dot':
            return torch.bmm(Q, K.transpose(1, 2))
        elif self.method == 'general':
            return torch.bmm(self.W(Q), K.transpose(1, 2))
        elif self.method == 'concat':
            B, L, _ = K.shape
            E = self.W(torch.cat((K, Q.repeat(1, L, 1)), dim=2))
            return torch.bmm(self.v.repeat(B, 1, 1), E.transpose(1, 2))


def get_inputs():
    return [torch.rand([4, 4, 4]), torch.rand([4, 4, 4]), torch.rand([4, 4, 4])
        ]


def get_init_inputs():
    return [[], {'H': 4}]
