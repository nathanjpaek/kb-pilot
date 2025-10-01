from _paritybench_helpers import _mock_config
import torch
import torch.nn as nn
from torch.nn import functional as F


class RWKV_TimeMix(nn.Module):

    def __init__(self, config, layer_id):
        super().__init__()
        assert config.n_attn % config.n_head == 0
        self.layer_id = layer_id
        self.ctx_len = config.ctx_len
        self.n_head = config.n_head
        self.head_size = config.n_attn // config.n_head
        self.time_ww = nn.Parameter(torch.ones(config.n_head, config.
            ctx_len, config.ctx_len))
        self.time_gamma = nn.Parameter(torch.ones(config.ctx_len, 1))
        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
        self.key = nn.Linear(config.n_embd, config.n_attn)
        self.value = nn.Linear(config.n_embd, config.n_attn)
        self.receptance = nn.Linear(config.n_embd, config.n_attn)
        self.output = nn.Linear(config.n_attn, config.n_embd)
        self.key.scale_init = 0
        self.receptance.scale_init = 0
        self.output.scale_init = 0

    def forward(self, x):
        B, T, C = x.size()
        x = torch.cat([self.time_shift(x[:, :, :C // 2]), x[:, :, C // 2:]],
            dim=-1)
        k = self.key(x)
        v = self.value(x)
        r = self.receptance(x)
        k = torch.clamp(k, max=30, min=-60)
        k = torch.exp(k)
        sum_k = torch.cumsum(k, dim=1)
        kv = (k * v).view(B, T, self.n_head, self.head_size)
        wkv = torch.einsum('htu,buhc->bthc', self.time_ww[:, :T, :T], kv
            ).contiguous().view(B, T, -1)
        rwkv = torch.sigmoid(r) * wkv / sum_k
        rwkv = self.output(rwkv)
        return rwkv * self.time_gamma[:T, :]


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


class Block(nn.Module):

    def __init__(self, config, layer_id):
        super().__init__()
        self.config = config
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.attn = RWKV_TimeMix(config, layer_id)
        self.mlp = RWKV_ChannelMix(config, layer_id)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


def get_inputs():
    return [torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'config': _mock_config(n_embd=4, n_attn=4, n_head=4,
        ctx_len=4, n_ffn=4), 'layer_id': 1}]
