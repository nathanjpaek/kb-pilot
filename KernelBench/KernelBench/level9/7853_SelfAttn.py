import torch
from torch import nn
from torch.nn import functional as F


class SelfAttn(nn.Module):

    def __init__(self, denc, dropout=0, enc_trans=None, dec_trans=None):
        super().__init__()
        self.scorer = nn.Linear(denc, 1)
        self.dropout = nn.Dropout(dropout)
        self.enc_trans = enc_trans or nn.Identity()

    def forward(self, enc, enc_mask):
        enc_trans = self.dropout(self.enc_trans(enc))
        raw_scores = self.scorer(enc).squeeze(2)
        scores = F.softmax(raw_scores - (1 - enc_mask) * 1e+20, dim=1)
        context = scores.unsqueeze(2).expand_as(enc_trans).sum(1)
        return context, scores


def get_inputs():
    return [torch.rand([4, 4, 4]), torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'denc': 4}]
