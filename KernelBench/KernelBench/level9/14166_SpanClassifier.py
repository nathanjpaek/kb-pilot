import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init


class SpanClassifier(nn.Module):

    def __init__(self, hidden_size: 'int', dropout_rate: 'float'):
        super(SpanClassifier, self).__init__()
        self.start_proj = nn.Linear(hidden_size, hidden_size)
        self.end_proj = nn.Linear(hidden_size, hidden_size)
        self.biaffine = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.concat_proj = nn.Linear(hidden_size * 2, 1)
        self.dropout = nn.Dropout(dropout_rate)
        self.reset_parameters()

    def forward(self, input_features):
        _bsz, seq_len, _dim = input_features.size()
        start_feature = self.dropout(F.gelu(self.start_proj(input_features)))
        end_feature = self.dropout(F.gelu(self.end_proj(input_features)))
        biaffine_logits = torch.bmm(torch.matmul(start_feature, self.
            biaffine), end_feature.transpose(1, 2))
        start_extend = start_feature.unsqueeze(2).expand(-1, -1, seq_len, -1)
        end_extend = end_feature.unsqueeze(1).expand(-1, seq_len, -1, -1)
        span_matrix = torch.cat([start_extend, end_extend], 3)
        concat_logits = self.concat_proj(span_matrix).squeeze(-1)
        return biaffine_logits + concat_logits

    def reset_parameters(self) ->None:
        init.kaiming_uniform_(self.biaffine, a=math.sqrt(5))


def get_inputs():
    return [torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'hidden_size': 4, 'dropout_rate': 0.5}]
