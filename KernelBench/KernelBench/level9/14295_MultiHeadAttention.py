import torch
import torch.nn as nn


def scaled_dot_product_attention(q, k, v, mask=None):
    """Calculate the attention weights.
    q, k, v must have matching leading dimensions.
    k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
    The mask has different shapes depending on its type(padding or look ahead)
    but it must be broadcastable for addition.

    Args:
    q: query shape == (..., seq_len_q, depth)
    k: key shape == (..., seq_len_k, depth)
    v: value shape == (..., seq_len_v, depth_v)
    mask: Float tensor with shape broadcastable
          to (..., seq_len_q, seq_len_k). Defaults to None.

    Returns:
    output, attention_weights
    """
    matmul_qk = torch.matmul(q, k.permute(0, 1, 3, 2))
    None
    dk = torch.tensor(k.size()[-1], dtype=torch.float32)
    scaled_attention_logits = matmul_qk / torch.sqrt(dk)
    if mask is not None:
        scaled_attention_logits += mask * -1000000000.0
    softmax = nn.Softmax(dim=-1)
    attention_weights = softmax(scaled_attention_logits)
    output = torch.matmul(attention_weights, v)
    return output, attention_weights


class MultiHeadAttention(nn.Module):

    def __init__(self, input_feature_size, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        assert d_model % self.num_heads == 0
        self.depth = d_model // self.num_heads
        self.dense = nn.Linear(input_feature_size, d_model)

    def split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
        """
        x = torch.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return x.permute(0, 2, 1, 3)

    def forward(self, v, k, q, mask, training=True):
        batch_size = q.size()[0]
        if q.size()[-2] > 10:
            pass
        else:
            pass
        q = q
        k = k
        v = v
        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)
        scaled_attention, attention_weights = scaled_dot_product_attention(q,
            k, v, mask)
        scaled_attention = scaled_attention.permute(0, 2, 1, 3)
        concat_attention = torch.reshape(scaled_attention, (batch_size, -1,
            self.d_model))
        output = self.dense(concat_attention)
        return output, attention_weights


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand(
        [4, 4, 4, 4]), torch.rand([4, 4, 16, 16])]


def get_init_inputs():
    return [[], {'input_feature_size': 4, 'd_model': 4, 'num_heads': 4}]
