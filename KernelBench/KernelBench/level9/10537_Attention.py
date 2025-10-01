import torch
import numpy as np


class Attention(torch.nn.Module):

    def __init__(self, d_model, heads):
        super().__init__()
        self.d_model = d_model
        self.heads = heads
        self.query = torch.nn.Linear(in_features=d_model, out_features=
            d_model, bias=False)
        self.key = torch.nn.Linear(in_features=d_model, out_features=
            d_model, bias=False)
        self.value = torch.nn.Linear(in_features=d_model, out_features=
            d_model, bias=False)
        self.dense = torch.nn.Linear(in_features=d_model, out_features=
            d_model, bias=False)

    def forward(self, x):
        assert x.shape[-1] == self.d_model
        seq_length = x.shape[0]
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        q = q.reshape(seq_length, self.heads, -1)
        k = k.reshape(seq_length, self.heads, -1)
        v = v.reshape(seq_length, self.heads, -1)
        q = q.transpose(0, 1)
        k = k.transpose(0, 1).transpose(1, 2)
        v = v.transpose(0, 1)
        logits = torch.matmul(q, k)
        logits = logits / np.sqrt(self.d_model // self.heads)
        logits = torch.softmax(logits, dim=-1)
        output = torch.matmul(logits, v)
        output = output.reshape((seq_length, self.d_model))
        output = self.dense(output)
        return output


def get_inputs():
    return [torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'d_model': 4, 'heads': 4}]
