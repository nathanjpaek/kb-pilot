import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.functional
import torch.nn.parallel
import torch.utils.data
import torch.optim
import torch.utils.data.distributed


class MultiHeadAttention(nn.Module):
    """
    input:
        query [N, T_q, query_dim]
        key   [N, T_k, key_dim]
    output:
        out   [N, T_q, num_units]
    """

    def __init__(self, query_dim, key_dim, num_units, num_heads):
        super(MultiHeadAttention, self).__init__()
        if key_dim % num_heads != 0:
            raise ValueError(
                'Key depth {} must be divisible by the number of attention heads {}'
                .format(key_dim, num_heads))
        self.num_units = num_units
        self.num_heads = num_heads
        self.query_scale = (key_dim // num_heads) ** -0.5
        self.query_linear = nn.Linear(query_dim, num_units, bias=False)
        self.key_linear = nn.Linear(key_dim, num_units, bias=False)
        self.value_linear = nn.Linear(key_dim, num_units, bias=False)

    def _split_heads(self, x):
        """
        Split x such to add an extra num_heads dimension
        Input:
            x: a Tensor with shape [batch_size, seq_len, depth]
        Returns:
            A Tensor with shape [batch_size, num_heads, seq_len, depth / num_heads]
        """
        if len(x.shape) != 3:
            raise ValueError('x must have rank 3')
        shape = x.shape
        return x.view(shape[0], shape[1], self.num_heads, shape[2] // self.
            num_heads).permute(0, 2, 1, 3)

    def _merge_heads(self, x):
        """
        Merge the extra num_heads into the last dimension
        Input:
            x: a Tensor with shape [batch_size, num_heads, seq_len, depth/num_heads]
        Output:
            A Tensor with shape [batch_size, seq_len, depth]
        """
        if len(x.shape) != 4:
            raise ValueError('x must have rank 4')
        shape = x.shape
        return x.permute(0, 2, 1, 3).contiguous().view(shape[0], shape[2], 
            shape[3] * self.num_heads)

    def forward(self, queries, keys, values):
        queries = self.query_linear(queries)
        keys = self.key_linear(keys)
        values = self.value_linear(values)
        queries = self._split_heads(queries)
        keys = self._split_heads(keys)
        values = self._split_heads(values)
        queries *= self.query_scale
        logits = torch.matmul(queries, keys.permute(0, 1, 3, 2))
        scores = F.softmax(logits, dim=-1)
        contexts = torch.matmul(scores, values)
        contexts = self._merge_heads(contexts)
        return contexts


def get_inputs():
    return [torch.rand([4, 4, 4]), torch.rand([4, 4, 4]), torch.rand([4, 4, 4])
        ]


def get_init_inputs():
    return [[], {'query_dim': 4, 'key_dim': 4, 'num_units': 4, 'num_heads': 4}]
