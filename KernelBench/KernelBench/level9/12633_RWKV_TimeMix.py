from _paritybench_helpers import _mock_config
import torch
import torch.nn as nn


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


def get_inputs():
    return [torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'config': _mock_config(n_attn=4, n_head=4, ctx_len=4,
        n_embd=4), 'layer_id': 1}]
