import torch
from torch import nn


def fill_with_neg_inf(t):
    """FP16-compatible function that fills a tensor with -inf."""
    return t.float().fill_(float('-inf')).type_as(t)


def buffered_future_mask(tensor1, tensor2, device):
    dim1 = dim2 = tensor1.size()
    if tensor2 is not None:
        dim2 = tensor2.size()
    future_mask = torch.triu(fill_with_neg_inf(torch.ones(dim1, dim2)), 1 +
        abs(dim2 - dim1))
    future_mask
    return future_mask[:dim1, :dim2]


class TransformerEncoderLayer(nn.Module):

    def __init__(self, embed_dim, num_heads=4, attn_dropout=0.1,
        relu_dropout=0.1, res_dropout=0.1, attn_mask=False):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.attn_dropout = attn_dropout
        self.self_attn = nn.MultiheadAttention(embed_dim=self.embed_dim,
            num_heads=self.num_heads, dropout=self.attn_dropout)
        self.attn_mask = attn_mask
        self.relu_dropout = nn.Dropout(p=relu_dropout)
        self.res_dropout = nn.Dropout(p=res_dropout)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(self.embed_dim, 4 * self.embed_dim)
        self.fc2 = nn.Linear(4 * self.embed_dim, self.embed_dim)
        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, x, x_k=None, x_v=None, src_key_padding_mask=None):
        residual = x
        x = self.layer_norm(x)
        mask = buffered_future_mask(x, x_k) if self.attn_mask else None
        if x_k is None and x_v is None:
            x, _ = self.self_attn(query=x, key=x, value=x, key_padding_mask
                =src_key_padding_mask, attn_mask=mask)
        else:
            x_k = self.layer_norm(x_k)
            x_v = self.layer_norm(x_v)
            x, _ = self.self_attn(query=x, key=x_k, value=x_v,
                key_padding_mask=src_key_padding_mask, attn_mask=mask)
        x = self.res_dropout(x)
        x = residual + x
        residual = x
        x = self.layer_norm(x)
        x = self.relu(self.fc1(x))
        x = self.relu_dropout(x)
        x = self.fc2(x)
        x = self.res_dropout(x)
        x = residual + x
        return x


def get_inputs():
    return [torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'embed_dim': 4}]
