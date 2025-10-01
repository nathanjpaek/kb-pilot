from _paritybench_helpers import _mock_config
import torch
import torch.nn as nn
from torch.nn import functional as F


class GeGLU(torch.nn.Module):

    def __init__(self, config, layer_id, time_shift=False):
        super().__init__()
        self.layer_id = layer_id
        if time_shift:
            self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
        hidden_sz = 3 * config.n_ffn
        self.key = nn.Linear(config.n_embd, hidden_sz)
        self.value = nn.Linear(config.n_embd, hidden_sz)
        self.weight = nn.Linear(hidden_sz, config.n_embd)

    def forward(self, x):
        _B, _T, C = x.size()
        if hasattr(self, 'time_shift'):
            x = torch.cat([self.time_shift(x[:, :, :C // 2]), x[:, :, C // 
                2:]], dim=-1)
        k = self.key(x)
        v = self.value(x)
        y = self.weight(F.gelu(k) * v)
        return y


def get_inputs():
    return [torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'config': _mock_config(n_ffn=4, n_embd=4), 'layer_id': 1}]
