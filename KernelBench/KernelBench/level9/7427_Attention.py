import torch
import torch.nn.functional as F


class Attention(torch.nn.Module):

    def __init__(self, features, attn_dim):
        super(Attention, self).__init__()
        self.to_q = torch.nn.Linear(features, attn_dim)
        self.to_k = torch.nn.Linear(features, attn_dim)
        self.to_v = torch.nn.Linear(features, attn_dim)
        self.project = torch.nn.Linear(attn_dim, features)

    def forward(self, x):
        Q = self.to_q(x)
        K = self.to_k(x)
        V = self.to_v(x)
        dots = torch.bmm(Q, K.permute(0, 2, 1))
        attn = F.softmax(dots, 0)
        out = torch.bmm(attn, V)
        out = self.project(out)
        return out


def get_inputs():
    return [torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'features': 4, 'attn_dim': 4}]
