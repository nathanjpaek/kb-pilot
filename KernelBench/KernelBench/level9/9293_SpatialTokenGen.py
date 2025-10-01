import torch
import torch.nn as nn


class SpatialTokenGen(nn.Module):

    def __init__(self, d_ffn, seq_len):
        super(SpatialTokenGen, self).__init__()
        self.layer_norm = nn.LayerNorm(d_ffn)
        self.squeeze_layer_i = nn.Linear(d_ffn, 1)
        self.squeeze_layer_ii = nn.Conv1d(seq_len, 1, 1)

    def forward(self, x):
        x = self.layer_norm(x)
        x = self.squeeze_layer_i(x)
        x = self.squeeze_layer_ii(x)
        tok = torch.mean(x)
        tok = torch.sigmoid(tok)
        return tok


def get_inputs():
    return [torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'d_ffn': 4, 'seq_len': 4}]
