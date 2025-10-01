import torch
import torch.nn as nn


class ConcatPositionalEncoding(nn.Module):

    def __init__(self, d_model=256, max_len=512):
        super().__init__()
        self.timing_table = nn.Parameter(torch.FloatTensor(max_len, d_model //
            2))
        nn.init.normal_(self.timing_table)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        timing = self.timing_table[None, :x.shape[1], :]
        x, timing = torch.broadcast_tensors(x, timing)
        out = torch.cat([x, timing], dim=-1)
        out = self.norm(out)
        return out


def get_inputs():
    return [torch.rand([4, 4, 4, 128])]


def get_init_inputs():
    return [[], {}]
