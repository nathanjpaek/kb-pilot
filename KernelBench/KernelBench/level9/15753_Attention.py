from _paritybench_helpers import _mock_config
from torch.nn import Module
import math
import torch
from torch import nn
from torch.nn import Parameter
from torch.nn.parameter import Parameter


class Conv1D(nn.Module):

    def __init__(self, nf, nx):
        super(Conv1D, self).__init__()
        self.nf = nf
        w = torch.empty(nx, nf)
        nn.init.normal_(w, std=0.02)
        self.weight = Parameter(w)
        self.bias = Parameter(torch.zeros(nf))

    def forward(self, x):
        size_out = x.size()[:-1] + (self.nf,)
        x.contiguous().view(-1, x.size(-1))
        x = torch.addmm(self.bias, x.contiguous().view(-1, x.size(-1)),
            self.weight)
        x = x.view(*size_out)
        return x


class Attention(Module):

    def __init__(self, nx, n_ctx, config, scale=False, can_be_stateful=False):
        super(Attention, self).__init__()
        n_state = nx
        assert n_state % config.n_head == 0
        self.register_buffer('bias', torch.tril(torch.ones(n_ctx, n_ctx)).
            view(1, 1, n_ctx, n_ctx))
        self.n_head = config.n_head
        self.split_size = n_state
        self.scale = scale
        self.c_attn = Conv1D(n_state * 3, nx)
        self.c_proj = Conv1D(n_state, nx)
        self.can_be_stateful = can_be_stateful
        self.attn_pdrop = nn.Dropout(config.attn_pdrop)
        if self.can_be_stateful:
            self.register_state('running_keys', torch.zeros((12, 0, 64)))
            self.register_state('running_values', torch.zeros((12, 0, 64)))

    def _attn(self, q, k, v, mask_self_attention):
        w = torch.matmul(q, k)
        if self.scale:
            w = w / math.sqrt(v.size(-1))
        if mask_self_attention is not None:
            w = w.masked_fill(mask_self_attention, -10000.0)
        w = nn.Softmax(dim=-1)(w)
        self.w = self.attn_pdrop(w)
        return torch.matmul(w, v)

    def merge_heads(self, x):
        x = x.permute(0, 2, 1, 3).contiguous()
        new_x_shape = x.size()[:-2] + (x.size(-2) * x.size(-1),)
        return x.view(*new_x_shape)

    def split_heads(self, x, k=False):
        new_x_shape = x.size()[:-1] + (self.n_head, x.size(-1) // self.n_head)
        x = x.view(*new_x_shape)
        if k:
            return x.permute(0, 2, 3, 1)
        else:
            return x.permute(0, 2, 1, 3)

    def forward(self, x, layer_past=None, mask_self_attention=None):
        x = self.c_attn(x)
        query, key, value = x.split(self.split_size, dim=2)
        query = self.split_heads(query)
        key = self.split_heads(key, k=True)
        value = self.split_heads(value)
        if self.can_be_stateful and self._is_stateful:
            self.running_keys = torch.cat([self.running_keys, key.transpose
                (-2, -1)], -2)
            key = self.running_keys.transpose(-2, -1)
            self.running_values = torch.cat([self.running_values, value], -2)
            value = self.running_values
        present = torch.stack((key.transpose(-2, -1), value))
        a = self._attn(query, key, value, mask_self_attention)
        a = self.merge_heads(a)
        a = self.c_proj(a)
        return a, present


def get_inputs():
    return [torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'nx': 4, 'n_ctx': 4, 'config': _mock_config(n_head=4,
        attn_pdrop=0.5)}]
