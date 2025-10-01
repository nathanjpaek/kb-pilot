import torch
import torch.nn.functional as F
import torch.nn as nn


class DotProductAttention(nn.Module):

    def __init__(self, k_dim):
        super(DotProductAttention, self).__init__()
        self.scale = 1.0 / k_dim ** 0.5

    def forward(self, hn, enc_out, mask=None):
        """
        :param hn: query - rnn的末隐层状态 [batch_size, hidden_size]
        :param enc_out: key - rnn的输出  [batch_size, seq_len, hidden_size]
        :param mask: [batch_size, seq_len] 0对应pad
        :return: att_out [batch_size, hidden_size]
        """
        att_score = torch.matmul(hn.unsqueeze(1), enc_out.transpose(1, 2)
            ).squeeze(1)
        att_score.mul_(self.scale)
        if mask is not None:
            att_score = att_score.masked_fill(~mask, -1000000000.0)
        att_weights = F.softmax(att_score, dim=1)
        att_out = torch.matmul(att_weights.unsqueeze(1), enc_out).squeeze(1)
        """
        # type 2
        # (bz, hidden_size, 1)
        hidden = hn.reshape(hn.size(0), -1, 1)  
        # (bz, n_step, hidden_size) * (bz, hidden_size, 1) -> (bz, n_step)
        att_score = torch.matmul(enc_out, hidden).squeeze(2) 
        att_score.mul_(self.scale)
        if mask is not None:
            att_score = att_score.masked_fill(~mask, -1e9)
        attn_weights = F.softmax(att_score, dim=1)
        # (bz, hidden_sze, n_step) * (bz, n_step, 1) -> (bz, hidden_size, 1)
        att_out = torch.matmul(enc_out.transpose(1, 2), attn_weights.unsqueeze(2)).squeeze(2)
        """
        return att_out


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'k_dim': 4}]
