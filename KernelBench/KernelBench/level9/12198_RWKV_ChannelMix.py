from _paritybench_helpers import _mock_config
import torch
import torch.nn as nn
from torch.nn import functional as F


class RWKV_ChannelMix(nn.Module):

    def __init__(self, config, layer_id):
        super().__init__()
        self.layer_id = layer_id
        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
        hidden_sz = 5 * config.n_ffn // 2
        self.key = nn.Linear(config.n_embd, hidden_sz)
        self.value = nn.Linear(config.n_embd, hidden_sz)
        self.weight = nn.Linear(hidden_sz, config.n_embd)
        self.receptance = nn.Linear(config.n_embd, config.n_embd)
        self.receptance.scale_init = 0
        self.weight.scale_init = 0

    def forward(self, x):
        _B, _T, C = x.size()
        x = torch.cat([self.time_shift(x[:, :, :C // 2]), x[:, :, C // 2:]],
            dim=-1)
        k = self.key(x)
        v = self.value(x)
        r = self.receptance(x)
        wkv = self.weight(F.mish(k) * v)
        rwkv = torch.sigmoid(r) * wkv
        return rwkv


def get_inputs():
    return [torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'config': _mock_config(n_ffn=4, n_embd=4), 'layer_id': 1}]
