import torch
from torch import Tensor
from typing import List
from typing import Tuple
from torch import nn
from functools import partial
from torch.nn.parameter import Parameter


class GroupedGRULayerMS(nn.Module):

    def __init__(self, in_ch: 'int', out_ch: 'int', n_freqs: 'int',
        n_groups: 'int', bias: 'bool'=True):
        super().__init__()
        assert n_freqs % n_groups == 0
        self.n_freqs = n_freqs
        self.g_freqs = n_freqs // n_groups
        self.n_groups = n_groups
        self.out_ch = self.g_freqs * out_ch
        self._in_ch = in_ch
        self.input_size = self.g_freqs * in_ch
        self.register_parameter('weight_ih_l', Parameter(torch.zeros(
            n_groups, 3 * self.out_ch, self.input_size), requires_grad=True))
        self.register_parameter('weight_hh_l', Parameter(torch.zeros(
            n_groups, 3 * self.out_ch, self.out_ch), requires_grad=True))
        if bias:
            self.register_parameter('bias_ih_l', Parameter(torch.zeros(
                n_groups, 3 * self.out_ch), requires_grad=True))
            self.register_parameter('bias_hh_l', Parameter(torch.zeros(
                n_groups, 3 * self.out_ch), requires_grad=True))
        else:
            self.bias_ih_l = None
            self.bias_hh_l = None

    def init_hidden(self, batch_size: 'int', device: 'torch.device'=torch.
        device('cpu')) ->Tensor:
        return torch.zeros(batch_size, self.n_groups, self.out_ch, device=
            device)

    def forward(self, input: 'Tensor', h=None) ->Tuple[Tensor, Tensor]:
        assert self.n_freqs == input.shape[-1]
        assert self._in_ch == input.shape[1]
        if h is None:
            h = self.init_hidden(input.shape[0])
        input = input.permute(0, 2, 3, 1).unflatten(2, (self.n_groups, self
            .g_freqs)).flatten(3)
        input = torch.einsum('btgi,goi->btgo', input, self.weight_ih_l)
        if self.bias_ih_l is not None:
            input = input + self.bias_ih_l
        h_out: 'List[Tensor]' = []
        for t in range(input.shape[1]):
            hh = torch.einsum('bgo,gpo->bgp', h, self.weight_hh_l)
            if self.bias_hh_l is not None:
                hh = hh + self.bias_hh_l
            ri, zi, ni = input[:, t].split(self.out_ch, dim=2)
            rh, zh, nh = hh.split(self.out_ch, dim=2)
            r = torch.sigmoid(ri + rh)
            z = torch.sigmoid(zi + zh)
            n = torch.tanh(ni + r * nh)
            h = (1 - z) * n + z * h
            h_out.append(h)
        out = torch.stack(h_out, dim=1)
        out = out.unflatten(3, (self.g_freqs, -1)).flatten(2, 3)
        out = out.permute(0, 3, 1, 2)
        return out, h


class GroupedGRUMS(nn.Module):

    def __init__(self, in_ch: 'int', out_ch: 'int', n_freqs: 'int',
        n_groups: 'int', n_layers: 'int'=1, bias: 'bool'=True, add_outputs:
        'bool'=False):
        super().__init__()
        self.n_layers = n_layers
        self.grus: 'List[GroupedGRULayerMS]' = nn.ModuleList()
        gru_layer = partial(GroupedGRULayerMS, out_ch=out_ch, n_freqs=
            n_freqs, n_groups=n_groups, bias=bias)
        self.gru0 = gru_layer(in_ch=in_ch)
        for _ in range(1, n_layers):
            self.grus.append(gru_layer(in_ch=out_ch))
        self.add_outputs = add_outputs

    def init_hidden(self, batch_size: 'int', device: 'torch.device'=torch.
        device('cpu')) ->Tensor:
        return torch.stack(tuple(self.gru0.init_hidden(batch_size, device) for
            _ in range(self.n_layers)))

    def forward(self, input: 'Tensor', h=None) ->Tuple[Tensor, Tensor]:
        if h is None:
            h = self.init_hidden(input.shape[0], input.device)
        h_out = []
        input, hl = self.gru0(input, h[0])
        h_out.append(hl)
        output = input
        for i, gru in enumerate(self.grus, 1):
            input, hl = gru(input, h[i])
            h_out.append(hl)
            if self.add_outputs:
                output = output + input
        if not self.add_outputs:
            output = input
        return output, torch.stack(h_out)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_ch': 4, 'out_ch': 4, 'n_freqs': 4, 'n_groups': 1}]
