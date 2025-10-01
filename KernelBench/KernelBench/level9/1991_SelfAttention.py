import torch
import torch.nn as nn


class SelfAttention(nn.Module):

    def __init__(self, embed_dims, heads):
        super(SelfAttention, self).__init__()
        self.heads = heads
        self.embed_dims = embed_dims
        self.depth = embed_dims // heads
        self.query = nn.Linear(self.depth, self.depth)
        self.key = nn.Linear(self.depth, self.depth)
        self.value = nn.Linear(self.depth, self.depth)
        self.fc_out = nn.Linear(self.depth * self.heads * 2, self.embed_dims)

    def forward(self, query, key, value, mask, isDecoder=False):
        batch, q_len, k_len, v_len = query.shape[0], query.shape[1], key.shape[
            1], value.shape[1]
        query = query.reshape(batch, q_len, self.heads, self.depth)
        key = key.reshape(batch, k_len, self.heads, self.depth)
        value = value.reshape(batch, v_len, self.heads, self.depth)
        query = self.query(query)
        key = self.key(key)
        value = self.value(value)
        energy = torch.einsum('bqhd, bkhd -> bhqk', [query, key])
        if isDecoder:
            None
            None
            None
            None
        if mask is not None:
            if isDecoder:
                None
            energy = energy.masked_fill(mask == 0, float('-1e20'))
        if isDecoder:
            None
            None
        energy = torch.softmax(energy / (self.depth ** 1 / 2), dim=-1)
        out = torch.einsum('bhqv, bvhd -> bqhd', [energy, value])
        out = out.reshape(batch, q_len, self.heads * self.depth)
        query = query.reshape(batch, q_len, self.heads * self.depth)
        out = torch.cat([query, out], dim=-1)
        out = self.fc_out(out)
        return out, energy


def get_inputs():
    return [torch.rand([4, 4, 4, 1]), torch.rand([4, 4, 4, 1]), torch.rand(
        [4, 4, 4, 1]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'embed_dims': 4, 'heads': 4}]
