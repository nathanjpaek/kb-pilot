import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ScaledDotProductAttention(nn.Module):

    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, query, key, value, mask=None):
        _1, _2, query_sequence_length, _3 = query.size()
        batch_size, num_head, key_sequence_length, size_per_head = key.size()
        query = query.view(batch_size, num_head, query_sequence_length,
            size_per_head)
        key = key.view(batch_size, num_head, size_per_head, key_sequence_length
            )
        attention_score = torch.einsum('abcd, abde -> abce', query, key)
        attention_score = attention_score / math.sqrt(size_per_head)
        if mask is not None:
            attention_score = attention_score.masked_fill(mask == 0, -
                1000000000.0)
        attention_score = F.softmax(attention_score, dim=-1)
        result = attention_score @ value
        return result, attention_score


class MultiHeadAttention(nn.Module):

    def __init__(self, model_dim, key_dim, value_dim, num_head):
        super(MultiHeadAttention, self).__init__()
        self.model_dim = model_dim
        self.key_dim = key_dim
        self.value_dim = value_dim
        self.num_head = num_head
        self.Wq = nn.Linear(model_dim, key_dim)
        self.Wk = nn.Linear(model_dim, key_dim)
        self.Wv = nn.Linear(model_dim, value_dim)
        self.attention = ScaledDotProductAttention()
        self.Wo = nn.Linear(value_dim, model_dim)

    def forward(self, query, key, value, mask=None):
        prj_query = self.Wq(query)
        prj_key = self.Wk(key)
        prj_value = self.Wv(value)
        multihead_query = self.multihead_split(prj_query)
        multihead_key = self.multihead_split(prj_key)
        multihead_value = self.multihead_split(prj_value)
        attention_output, _attention_score = self.attention(multihead_query,
            multihead_key, multihead_value, mask=mask)
        output = self.multihead_concat(attention_output)
        output = self.Wo(output)
        return output

    def multihead_split(self, tensor):
        batch_size, sequence_length, hidden_size = tensor.size()
        size_per_head = hidden_size // self.num_head
        return tensor.view(batch_size, self.num_head, sequence_length,
            size_per_head)

    def multihead_concat(self, tensor):
        batch_size, num_head, sequence_length, size_per_head = tensor.size()
        hidden_size = num_head * size_per_head
        return tensor.view(batch_size, sequence_length, hidden_size)


class FeedForward(nn.Module):

    def __init__(self, model_dim, hidden_dim, drop_prob):
        super(FeedForward, self).__init__()
        self.model_dim = model_dim
        self.hidden_dim = hidden_dim
        self.drop_prob = drop_prob
        self.linearlayer1 = nn.Linear(model_dim, hidden_dim)
        self.linearlayer2 = nn.Linear(hidden_dim, model_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(drop_prob)

    def forward(self, tensor):
        tensor = self.dropout(self.relu(self.linearlayer1(tensor)))
        return self.linearlayer2(tensor)


class EncoderLayer(nn.Module):

    def __init__(self, model_dim, key_dim, value_dim, hidden_dim, num_head,
        drop_prob):
        super(EncoderLayer, self).__init__()
        self.attention = MultiHeadAttention(model_dim, key_dim, value_dim,
            num_head)
        self.normalization1 = nn.LayerNorm(model_dim)
        self.dropout1 = nn.Dropout(drop_prob)
        self.ffn = FeedForward(model_dim, hidden_dim, drop_prob)
        self.normalization2 = nn.LayerNorm(model_dim)
        self.dropout2 = nn.Dropout(drop_prob)

    def forward(self, tensor, source_mask):
        residual = tensor
        tensor = self.attention(query=tensor, key=tensor, value=tensor,
            mask=source_mask)
        tensor = self.dropout1(self.normalization1(tensor + residual))
        residual = tensor
        tensor = self.ffn(tensor)
        tensor = self.dropout2(self.normalization2(tensor + residual))
        return tensor


def get_inputs():
    return [torch.rand([4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'model_dim': 4, 'key_dim': 4, 'value_dim': 4, 'hidden_dim':
        4, 'num_head': 4, 'drop_prob': 0.5}]
