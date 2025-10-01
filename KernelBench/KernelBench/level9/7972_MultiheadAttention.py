import math
import torch
import torch.nn as nn
import torch.utils.data


class MultiheadAttention(nn.Module):
    """
  Multihead attention mechanism (dot attention)
  """

    def __init__(self, num_hidden_k, dropout_p=0.1):
        """
    :param num_hidden_k: dimension of hidden 
    """
        super(MultiheadAttention, self).__init__()
        self.num_hidden_k = num_hidden_k
        self.attn_dropout = nn.Dropout(p=dropout_p)

    def forward(self, key, value, query, mask=None):
        attn = torch.matmul(query, key.transpose(2, 3))
        attn = attn / math.sqrt(self.num_hidden_k)
        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1000000000.0)
        attn = torch.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)
        result = torch.matmul(attn, value)
        return result, attn


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand(
        [4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'num_hidden_k': 4}]
