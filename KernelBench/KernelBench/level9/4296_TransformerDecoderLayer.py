import torch
from torch import nn
import torch.nn.functional as F


def _get_activation_fn(activation):
    if activation == 'relu':
        return F.relu
    raise RuntimeError('activation shud be relu, not {}'.format(activation))


class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.3,
        activation='relu'):
        super(TransformerDecoderLayer, self).__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.nhead = nhead
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.masked_attn = nn.MultiheadAttention(d_model, nhead, dropout=
            dropout)
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.activation = _get_activation_fn(activation)

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(TransformerDecoderLayer, self).__setstate__(state)

    def forward(self, input, output, mask):
        x = self.norm1(input)
        x = torch.transpose(x, 0, 1)
        x = self.masked_attn(x, x, x, attn_mask=mask)[0]
        x = torch.transpose(x, 0, 1)
        input = input + self.dropout1(x)
        x = self.norm2(input)
        x = torch.transpose(x, 0, 1)
        output = torch.transpose(output, 0, 1)
        x = self.attn(x, output, output)[0]
        output = torch.transpose(output, 0, 1)
        x = torch.transpose(x, 0, 1)
        input = input + self.dropout2(x)
        x = self.norm3(input)
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        input = input + self.dropout3(x)
        return input


def get_inputs():
    return [torch.rand([4, 4]), torch.rand([4, 4]), torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'d_model': 4, 'nhead': 4, 'dim_feedforward': 4}]
