import torch
import torch.nn as nn


class TransformerBlock(nn.Module):

    def __init__(self, max_len, hidden_size, hidden_dropout,
        attention_heads, feed_forward_size):
        super().__init__()
        self.pre_layer_norm_1 = nn.LayerNorm([max_len, hidden_size])
        self.dropout_1 = nn.Dropout(p=hidden_dropout)
        self.multi_head_attention = nn.MultiheadAttention(hidden_size,
            attention_heads, hidden_dropout)
        self.pre_layer_norm_2 = nn.LayerNorm([max_len, hidden_size])
        self.dropout_2 = nn.Dropout(p=hidden_dropout)
        self.feed_forward_1 = nn.Linear(hidden_size, feed_forward_size)
        self.feed_forward_2 = nn.Linear(feed_forward_size, hidden_size)
        self.activation = nn.GELU()

    def forward(self, x, p):
        x_ = self.pre_layer_norm_1(x)
        x_ = self.dropout_1(x_)
        x_ = x_.view([x_.shape[1], x_.shape[0], x_.shape[2]])
        x_ = self.multi_head_attention(x_, x_, x_)[0]
        x_ = x_.view([x_.shape[1], x_.shape[0], x_.shape[2]])
        x = (x + x_) * (1 / p)
        x_ = self.pre_layer_norm_2(x)
        x_ = self.dropout_2(x_)
        x_ = self.feed_forward_1(x_)
        x_ = self.feed_forward_2(x_)
        x_ = self.activation(x_)
        x_ = (x + x_) * (1 / p)
        return x_


def get_inputs():
    return [torch.rand([4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'max_len': 4, 'hidden_size': 4, 'hidden_dropout': 0.5,
        'attention_heads': 4, 'feed_forward_size': 4}]
