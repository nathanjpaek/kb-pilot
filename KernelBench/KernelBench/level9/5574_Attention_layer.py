import math
import torch
import torch.nn as nn


def calculate_attention(query: 'torch.Tensor', key: 'torch.Tensor', value:
    'torch.Tensor', mask: 'torch.Tensor'):
    """Calclulate Attention 
    @param:
        query: torch.Tensor (Batch_size, max_seq_len, hidden_size)
        key: torch.Tensor (Batch_size, max_seq_len, hidden_size)
        value : torch.Tensor(Batch_size, max_seq_len, word_vector_size)
        mask : torch.Tensor.Boolean(Batch_size, max_seq_len);
    """
    hidden_size = query.size(-1)
    max_seq_size = mask.size(-1)
    scores = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(hidden_size
        )
    mask_ = mask.unsqueeze(-1).expand(-1, -1, max_seq_size)
    scores = scores.masked_fill(mask_ == 0, -1000000000.0)
    attention_matrix = torch.softmax(scores, dim=-1)
    output = torch.matmul(attention_matrix, value)
    return output


class Attention_layer(nn.Module):
    """Attention Unit"""

    def __init__(self, kernel_input: 'int', kernel_output: 'int', dropout:
        'int'=0.2):
        super(Attention_layer, self).__init__()
        self.key_kernel = nn.Linear(kernel_input, kernel_output)
        self.query_kernel = nn.Linear(kernel_input, kernel_output)
        self.value_kernel = nn.Linear(kernel_input, kernel_output)
        self.normalize_kernel = nn.Linear(kernel_output, 1, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: 'torch.Tensor', mask: 'torch.Tensor'):
        """
        @params:
            x: (batch_size, max_seq_len, word_emb_size)
            mask: (batch_size, max_seq_len)
        """
        key = self.key_kernel(x)
        query = self.query_kernel(x)
        value = self.value_kernel(x)
        att_output = calculate_attention(query, key, value, mask)
        att_output = self.dropout(att_output)
        att_output = self.normalize_kernel(att_output)
        att_output = torch.softmax(att_output, -1)
        return att_output


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'kernel_input': 4, 'kernel_output': 4}]
