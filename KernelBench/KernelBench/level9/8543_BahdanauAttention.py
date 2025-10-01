import torch
import torch.nn as nn


class BahdanauAttention(nn.Module):

    def __init__(self, hidden_dim):
        super(BahdanauAttention, self).__init__()
        self.W = nn.Linear(hidden_dim, hidden_dim)
        self.U = nn.Linear(hidden_dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1)

    def forward(self, dec_h_prev, enc_h_all, epsilon=1e-08):
        w = self.W(dec_h_prev).unsqueeze(1)
        u = self.U(enc_h_all)
        s = self.v(torch.tanh(w + u))
        m, _ = s.max(dim=1, keepdim=True)
        s = torch.exp(s - m)
        a = s / (torch.sum(s, dim=1, keepdim=True) + epsilon)
        c = torch.sum(enc_h_all * a, dim=1)
        return c, a


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'hidden_dim': 4}]
