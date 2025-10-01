from torch.nn import Module
import torch
from torch.nn import Linear
from typing import Optional
from torch.nn import Tanh


def masked_softmax(vector: 'torch.FloatTensor', mask: 'torch.ByteTensor'):
    """
    计算带有 masked 的 softmax
    :param vector: shape: (B, seq_len)
    :param mask: shape: (B, seq_len),
    :return: (B, seq_len)
    """
    exp_vector = vector.exp()
    masked_vector = exp_vector * mask.float()
    return masked_vector / torch.sum(masked_vector, dim=-1, keepdim=True)


class AttentionSeq2Vec(Module):
    """
    基于 attention 将 seq2vec. 具体操作如下:

    1. sequence: (B, seq_len, input_size)
    2. K = WkSeqeunce 将 sequence 进行变换, K shape: (B, seq_len, query_hidden_size)
    3. Q = Shape: (query_hidden_size)
    4. attention = softmax(KQ), shape: (B, seq_len)
    5. V = WvSequence, shape: (B, seq_len, value_hidden_size); 如果 value_hidden_size is None,
    shape: (B, seq_len, input_size)
    6. sum(V*attention, dim=-1), shape: (B, input_size)
    """

    def __init__(self, input_size: 'int', query_hidden_size: 'int',
        value_hidden_size: 'Optional[int]'=None):
        """
        初始化。遵循 Q K V，计算 attention 方式。
        :param input_size: 输入的 sequence token 的 embedding dim
        :param query_hidden_size: 将 seqence 变成 Q 的时候，变换后的 token embedding dim.
        :param value_hidden_size: 将 seqence 变成 V 的时候, 变换后的 token embedding dim.
        如果 value_hidden_size is None, 那么，该模型就与 2016-Hierarchical Attention Networks for Document Classification
        是一致的, 最后的输出结果 shape (B, seq_len, input_size);
        如果 value_hidden_size 被设置了, 那么，就与 Attention is All your Need 中 变换是一致的, 最后的输出结果
        shape (B, seq_len, value_hidden_size)
        """
        super().__init__()
        self.wk = Linear(in_features=input_size, out_features=
            query_hidden_size, bias=True)
        self.key_activation = Tanh()
        self.attention = Linear(in_features=query_hidden_size, out_features
            =1, bias=False)
        self.wv = None
        if value_hidden_size is not None:
            self.wv = Linear(in_features=input_size, out_features=
                value_hidden_size, bias=True)
        self.reset_parameters()

    def reset_parameters(self):
        pass

    def forward(self, sequence: 'torch.LongTensor', mask:
        'Optional[torch.ByteTensor]') ->torch.FloatTensor:
        """
        执行 attetion seq2vec
        :param sequence: 输入的token 序列, shape: (batch_size, seq_len, input_size)
        :param mask: mask shape: (batch_size, seq_len)
        :return: attention 编码向量, shape: (batch_size, value_hidden_size or input_size)
        """
        assert sequence.dim(
            ) == 3, 'sequence shape: (batch_size, seq_len, input_size)'
        if mask is not None:
            assert mask.dim() == 2, 'mask shape: (batch_size, seq_len)'
        key = self.wk(sequence)
        key = self.key_activation(key)
        attention = self.attention(key)
        attention = torch.squeeze(attention, dim=-1)
        if mask is not None:
            attention = masked_softmax(vector=attention, mask=mask)
        else:
            attention = torch.softmax(attention, dim=-1)
        if self.wv is not None:
            value = self.wv(sequence)
        else:
            value = sequence
        attentioned_value = value * attention.unsqueeze(dim=-1)
        vector = torch.sum(attentioned_value, dim=1)
        return vector


def get_inputs():
    return [torch.rand([4, 4, 4]), torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'input_size': 4, 'query_hidden_size': 4}]
