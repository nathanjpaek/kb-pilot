import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class Attn(nn.Module):

    def __init__(self, hidden):
        super().__init__()
        self.hidden = hidden
        self.attn = nn.Linear(self.hidden * 2, hidden)
        self.v = nn.Parameter(torch.rand(hidden))
        stdv = 1.0 / math.sqrt(self.v.size(0))
        self.v.data.normal_(mean=0, std=stdv)

    def forward(self, hidden, enc_out):
        l = enc_out.size(0)
        enc_out.size(1)
        H = hidden.repeat(l, 1, 1).transpose(0, 1)
        enc_out = enc_out.transpose(0, 1)
        attn_score = self.score(H, enc_out)
        return F.softmax(attn_score, dim=1).unsqueeze(1)

    def score(self, hidden, enc_out):
        """
        concat score function
        score(s_t, h_i) = vT_a tanh(Wa[s_t; h_i])
        """
        energy = torch.tanh(self.attn(torch.cat([hidden, enc_out], 2)))
        energy = energy.transpose(2, 1)
        v = self.v.repeat(enc_out.data.shape[0], 1).unsqueeze(1)
        energy = torch.bmm(v, energy)
        return energy.squeeze(1)


def get_inputs():
    return [torch.rand([4, 4]), torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'hidden': 4}]
