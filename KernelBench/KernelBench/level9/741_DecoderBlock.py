import torch
from torch import nn


class DecoderBlock(nn.Module):
    """
    A block in decoder that makes use of sentence representation
    TODO: block is a boring name; there gotta be a more creative name for this step
    """

    def __init__(self, d_model, dropout=0.1, mode='add_attn'):
        super().__init__()
        assert mode in ('add_attn', 'cat_attn')
        self.mode = mode
        if mode == 'add_attn':
            self.w1 = nn.Linear(d_model, d_model)
            self.w2 = nn.Linear(d_model, d_model)
        elif mode == 'cat_attn':
            self.w = nn.Linear(d_model + d_model, d_model)
        else:
            raise Exception()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sent_repr):
        if self.mode == 'add_attn':
            scores = self.w1(x) + self.w2(sent_repr)
        elif self.mode == 'cat_attn':
            scores = self.w(torch.cat([x, sent_repr], dim=-1))
        else:
            raise Exception()
        weights = scores.sigmoid()
        weights = self.dropout(weights)
        return sent_repr * weights


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'d_model': 4}]
