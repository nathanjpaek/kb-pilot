import torch
import torch.nn.functional as F
import torch.nn as nn


class AdditiveAttention(nn.Module):

    def __init__(self, k_size, v_size, hidden_size=None, bias=True):
        super(AdditiveAttention, self).__init__()
        if hidden_size is None:
            hidden_size = v_size
        self.W1 = nn.Linear(k_size, hidden_size, bias=False)
        self.W2 = nn.Linear(v_size, hidden_size, bias=bias)
        self.V = nn.Linear(hidden_size, 1, bias=False)
        self.reset_params()

    def reset_params(self):
        nn.init.xavier_normal_(self.W1.weight)
        nn.init.xavier_normal_(self.W2.weight)
        nn.init.xavier_uniform_(self.V.weight)

    def forward(self, q, v, mask=None):
        """
        :param q: (bz, hidden_size)
        :param v: (bz, n_step, hidden_size)
        :param mask: (bz, n_step)
        :return:
        """
        expand_q = q.unsqueeze(1)
        att_score = self.V(torch.tanh(self.W1(expand_q) + self.W2(v)))
        if mask is not None:
            att_score = att_score.masked_fill(~mask.unsqueeze(-1), -
                1000000000.0)
        att_weights = F.softmax(att_score, dim=1)
        attn_dist = att_weights.squeeze(dim=-1)
        att_out = (att_weights * v).sum(dim=1)
        return att_out, attn_dist


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'k_size': 4, 'v_size': 4}]
