import torch
import torch.nn as nn


class NNAttention(nn.Module):

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.q_net = nn.Linear(in_dim, out_dim)
        self.k_net = nn.Linear(in_dim, out_dim)
        self.v_net = nn.Linear(in_dim, out_dim)

    def forward(self, Q, K, V):
        q = self.q_net(Q)
        k = self.k_net(K)
        v = self.v_net(V)
        attn = torch.einsum('ijk,ilk->ijl', q, k)
        attn = attn
        attn_prob = torch.softmax(attn, dim=-1)
        attn_vec = torch.einsum('ijk,ikl->ijl', attn_prob, v)
        return attn_vec


def get_inputs():
    return [torch.rand([4, 4, 4]), torch.rand([4, 4, 4]), torch.rand([4, 4, 4])
        ]


def get_init_inputs():
    return [[], {'in_dim': 4, 'out_dim': 4}]
