import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def scaled_self_attention(q, k, v, key_size):
    weight = torch.matmul(q, k)
    weight = F.softmax(weight / math.sqrt(key_size), dim=-1)
    attention = torch.matmul(weight, v)
    return attention


class Multihead_Attention_Layer(nn.Module):

    def __init__(self, embedding, heads):
        super(Multihead_Attention_Layer, self).__init__()
        self.embedding = embedding
        self.heads = heads
        assert self.embedding % self.heads == 0, 'The number of embedding channels must be divisible by the number of heads'
        self.head_dim = self.embedding // self.heads
        self.embedding_layer = nn.Conv1d(self.embedding, self.embedding * 3,
            kernel_size=1, bias=False)
        self.out = nn.Conv1d(self.embedding, self.embedding, kernel_size=1,
            bias=False)

    def forward(self, x):
        batch_size, channels, num_points = x.size()
        qkv = self.embedding_layer(x)
        qkv = qkv.permute(0, 2, 1)
        qkv = qkv.reshape(batch_size, num_points, self.heads, 3 * self.head_dim
            )
        qkv = qkv.permute(0, 2, 1, 3)
        q, k, v = torch.chunk(qkv, 3, dim=-1)
        k = k.permute(0, 1, 3, 2)
        values = scaled_self_attention(q, k, v, self.embedding / self.heads)
        values = values.permute(0, 2, 1, 3)
        values = values.reshape(batch_size, num_points, channels)
        x = values.permute(0, 2, 1)
        x = self.out(x)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'embedding': 4, 'heads': 4}]
