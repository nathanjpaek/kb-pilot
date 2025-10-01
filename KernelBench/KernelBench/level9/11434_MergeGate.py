import torch
import torch.nn as nn
import torch.nn.functional as F


class MergeGate(nn.Module):

    def __init__(self, hidden_size):
        super(MergeGate, self).__init__()
        self.hidden_size = hidden_size
        self.WSh = nn.Linear(hidden_size, hidden_size)
        self.WSc = nn.Linear(hidden_size, hidden_size)
        self.WSr = nn.Linear(hidden_size, hidden_size)
        self.wS = nn.Linear(hidden_size, 1)

    def forward(self, attn_applied_c, attn_applied_r, hidden):
        content_c = self.WSc(attn_applied_c) + self.WSh(hidden.transpose(0, 1))
        score_c = self.wS(F.tanh(content_c))
        content_r = self.WSr(attn_applied_r) + self.WSh(hidden.transpose(0, 1))
        score_r = self.wS(F.tanh(content_r))
        gama_t = F.sigmoid(score_c - score_r)
        c_t = gama_t * attn_applied_c + (1 - gama_t) * attn_applied_r
        return c_t


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand(
        [4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'hidden_size': 4}]
