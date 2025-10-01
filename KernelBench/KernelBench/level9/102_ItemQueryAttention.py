import torch
import torch as t
import torch.nn as nn


class ItemQueryAttention(nn.Module):
    """
    基于项的注意力机制。使用查询集序列对支持集的样本序列进行注意力对齐，
    得到一个支持集样本的注意力上下文向量。由于注意力向量不依赖于RNN的
    上下文向量，因此该注意力属于基于项的注意力，可以并行化处理
    """

    def __init__(self, feature_size, hidden_size):
        super(ItemQueryAttention, self).__init__()
        self.W = nn.Linear(feature_size, hidden_size)

    def forward(self, qs, hs):
        assert len(qs.size()) == 3 and len(hs.size()) == 3, '输入attention的尺寸不符！'
        s_size = hs.size(0)
        q_size = qs.size(0)
        feature_size = qs.size(2)
        seq_size = hs.size(1)
        qs = qs.repeat((s_size, 1, 1, 1)).transpose(0, 1).contiguous(
            ).unsqueeze(2).repeat(1, 1, seq_size, 1, 1).transpose(2, 3)
        hs = hs.repeat((q_size, 1, 1, 1)).unsqueeze(2).repeat(1, 1,
            seq_size, 1, 1)
        att = t.sum(t.tanh(self.W(qs) * self.W(hs)), dim=4).softmax(dim=3
            ).squeeze()
        att = att.unsqueeze(dim=4).repeat((1, 1, 1, 1, feature_size))
        hs = (att * hs).sum(dim=3)
        return hs


def get_inputs():
    return [torch.rand([4, 4, 4]), torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'feature_size': 4, 'hidden_size': 4}]
