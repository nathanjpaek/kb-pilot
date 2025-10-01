import torch
import torch.utils.data
import torch.nn as nn


class SoftAttention(torch.nn.Module):
    """
    v = tanh(hW + b)
    w = softmax(v*u)
    out = sum wh

    see eqs 5-7 in https://www.sciencedirect.com/science/article/abs/pii/S0924271619300115
    """

    def __init__(self, hidden_dim):
        super(SoftAttention, self).__init__()
        self.linear = nn.Linear(in_features=hidden_dim, out_features=
            hidden_dim, bias=True)
        self.tanh = nn.Tanh()
        self.ua = nn.Parameter(torch.Tensor(hidden_dim))
        torch.nn.init.normal_(self.ua, mean=0.0, std=0.1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        N, *_ = x.shape
        va = self.tanh(self.linear(x))
        batchwise_ua = self.ua.repeat(N, 1)
        omega = self.softmax(torch.bmm(va, batchwise_ua.unsqueeze(-1)))
        rnn_feat = torch.bmm(x.transpose(1, 2), omega).squeeze(-1)
        return rnn_feat


def get_inputs():
    return [torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'hidden_dim': 4}]
