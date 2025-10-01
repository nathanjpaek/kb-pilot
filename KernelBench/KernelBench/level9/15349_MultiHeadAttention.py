import torch
import numpy as np


class MultiHeadAttention(torch.nn.Module):

    def __init__(self, input_size, output_size, num_heads,
        output_attentions=False):
        super(MultiHeadAttention, self).__init__()
        self.output_attentions = output_attentions
        self.num_heads = num_heads
        self.d_model_size = input_size
        self.depth = int(output_size / self.num_heads)
        self.Wq = torch.nn.Linear(input_size, output_size)
        self.Wk = torch.nn.Linear(input_size, output_size)

    def split_into_heads(self, x, batch_size):
        x = x.reshape(batch_size, -1, self.num_heads, self.depth)
        return x.permute([0, 2, 1, 3])

    def forward(self, k, q):
        batch_size = q.shape[0]
        q = self.Wq(q)
        k = self.Wk(k)
        q = self.split_into_heads(q, batch_size)
        k = self.split_into_heads(k, batch_size)
        attn_score = torch.matmul(q, k.permute(0, 1, 3, 2))
        attn_score = attn_score / np.sqrt(k.shape[-1])
        return attn_score


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'input_size': 4, 'output_size': 4, 'num_heads': 4}]
