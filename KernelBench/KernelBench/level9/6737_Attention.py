import torch
import torch.nn as nn


class Attention(nn.Module):

    def __init__(self):
        super().__init__()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, Q, K, V, mask=None, dk=64):
        w = torch.bmm(Q, K.transpose(1, 2))
        if mask is not None:
            assert w.size() == mask.size()
            w.masked_fill_(mask, -float('inf'))
        w = self.softmax(w / dk ** 0.5)
        c = torch.bmm(w, V)
        return c


def get_inputs():
    return [torch.rand([4, 4, 4]), torch.rand([4, 4, 4]), torch.rand([4, 4, 4])
        ]


def get_init_inputs():
    return [[], {}]
