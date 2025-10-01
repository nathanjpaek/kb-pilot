import torch
import torch.nn as nn
import torch.nn.functional as F


class DotAttention(nn.Module):

    def __init__(self, dropout=0.0):
        super(DotAttention, self).__init__()
        self.dropout = dropout

    def forward(self, Q, K, V, mask_out=None, head_mask=None):
        """
        一般输入信息 X 时，假设 K = V = X

        att_weight = softmax( score_func(q, k) )
        att = sum( att_weight * v )

        :param Q: [..., L, H]
        :param K: [..., S, H]
        :param V: [..., S, H]
        :param mask_out: [..., 1, S]
        :return:
        """
        H = Q.size(-1)
        scale = float(H) ** 0.5
        attention_weight = torch.matmul(Q, K.transpose(-1, -2)) / scale
        if mask_out is not None:
            while mask_out.dim() != Q.dim():
                mask_out = mask_out.unsqueeze(1)
            attention_weight.masked_fill_(mask_out, -100000000.0)
        attention_weight = F.softmax(attention_weight, dim=-1)
        attention_weight = F.dropout(attention_weight, self.dropout)
        if head_mask is not None:
            attention_weight = attention_weight * head_mask
        attention_out = torch.matmul(attention_weight, V)
        return attention_out, attention_weight


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand(
        [4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
