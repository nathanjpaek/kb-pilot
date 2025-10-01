import torch
from torch import nn
import torch as t
from torch.nn import functional as F


def getMaskFromLens(lens, max_seq_len=200, expand_feature_dim=None):
    if type(lens) == list:
        lens = t.LongTensor(lens)
    batch_size = len(lens)
    idx_matrix = t.arange(0, max_seq_len, 1).repeat((batch_size, 1))
    len_mask = lens.unsqueeze(1)
    mask = idx_matrix.ge(len_mask)
    if expand_feature_dim is not None:
        mask = mask.unsqueeze(-1).repeat_interleave(expand_feature_dim, dim=-1)
    return mask


class BiliAttnReduction(nn.Module):

    def __init__(self, input_dim, max_seq_len=200, **kwargs):
        super(BiliAttnReduction, self).__init__()
        self.MaxSeqLen = max_seq_len
        self.IntAtt = nn.Linear(input_dim, input_dim, bias=False)
        self.ExtAtt = nn.Linear(input_dim, 1, bias=False)

    def forward(self, x, lens=None):
        if isinstance(x, t.nn.utils.rnn.PackedSequence):
            x, lens = nn.utils.rnn.pad_packed_sequence(x, batch_first=True)
        feature_dim = x.size(2)
        att_weight = self.ExtAtt(t.tanh(self.IntAtt(x))).squeeze()
        if lens is not None:
            if not isinstance(lens, t.Tensor):
                lens = t.Tensor(lens)
            mask = getMaskFromLens(lens, self.MaxSeqLen)
            att_weight.masked_fill_(mask, float('-inf'))
        att_weight = t.softmax(att_weight, dim=1).unsqueeze(-1).repeat((1, 
            1, feature_dim))
        return (att_weight * x).sum(dim=1)

    @staticmethod
    def static_forward(x, params, lens=None):
        if isinstance(x, t.nn.utils.rnn.PackedSequence):
            x, lens = nn.utils.rnn.pad_packed_sequence(x, batch_first=True)
        feature_dim = x.size(2)
        att_weight = F.linear(input=t.tanh(F.linear(input=x, weight=params[
            0])), weight=params[1]).squeeze()
        if lens is not None:
            if not isinstance(lens, t.Tensor):
                lens = t.Tensor(lens)
            mask = getMaskFromLens(lens)
            att_weight.masked_fill_(mask, float('-inf'))
        att_weight = t.softmax(att_weight, dim=1).unsqueeze(-1).repeat((1, 
            1, feature_dim))
        return (att_weight * x).sum(dim=1)


def get_inputs():
    return [torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'input_dim': 4}]
