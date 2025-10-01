import torch
import numpy as np


def scaled_dot_product_attention(q, k, v, mask):
    matmul_qk = torch.matmul(q, k.permute(0, 1, 3, 2))
    dk = k.shape[-1]
    scaled_attention_logits = matmul_qk / np.sqrt(dk)
    if mask is not None:
        scaled_attention_logits += mask * -1000000000.0
    attention_weights = torch.softmax(scaled_attention_logits, dim=-1)
    output = torch.matmul(attention_weights, v)
    return output


class MultiHeadAttention(torch.nn.Module):

    def __init__(self, d_model_size, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model_size = d_model_size
        self.depth = int(d_model_size / self.num_heads)
        self.Wq = torch.nn.Linear(d_model_size, d_model_size)
        self.Wk = torch.nn.Linear(d_model_size, d_model_size)
        self.Wv = torch.nn.Linear(d_model_size, d_model_size)
        self.dense = torch.nn.Linear(d_model_size, d_model_size)

    def split_into_heads(self, x, batch_size):
        x = x.reshape(batch_size, -1, self.num_heads, self.depth)
        return x.permute([0, 2, 1, 3])

    def forward(self, v, k, q, mask):
        batch_size = q.shape[0]
        q = self.Wq(q)
        k = self.Wk(k)
        v = self.Wv(v)
        q = self.split_into_heads(q, batch_size)
        k = self.split_into_heads(k, batch_size)
        v = self.split_into_heads(v, batch_size)
        scaled_attention = scaled_dot_product_attention(q, k, v, mask).permute(
            [0, 2, 1, 3])
        original_size_attention = scaled_attention.reshape(batch_size, -1,
            self.d_model_size)
        output = self.dense(original_size_attention)
        return output


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand(
        [4, 4, 4, 4]), torch.rand([4, 4, 16, 16])]


def get_init_inputs():
    return [[], {'d_model_size': 4, 'num_heads': 4}]
