import math
import torch
import torch.nn as nn
import torch.utils.data
from itertools import product as product
from math import sqrt as sqrt


class SpatialGate(nn.Module):

    def __init__(self, in_channels: 'int', num_groups: 'int'=1, kernel_size:
        'int'=1, padding: 'int'=0, stride: 'int'=1, gate_activation: 'str'=
        'ReTanH', gate_activation_kargs: 'dict'=None, get_running_cost:
        'callable'=None):
        super(SpatialGate, self).__init__()
        self.num_groups = num_groups
        self.gate_conv = nn.Conv2d(in_channels, num_groups, kernel_size,
            padding=padding, stride=stride)
        self.gate_activation = gate_activation
        self.gate_activation_kargs = gate_activation_kargs
        if gate_activation == 'ReTanH':
            self.gate_activate = lambda x: torch.tanh(x).clamp(min=0)
        elif gate_activation == 'Sigmoid':
            self.gate_activate = lambda x: torch.sigmoid(x)
        elif gate_activation == 'GeReTanH':
            assert 'tau' in gate_activation_kargs
            tau = gate_activation_kargs['tau']
            ttau = math.tanh(tau)
            self.gate_activate = lambda x: ((torch.tanh(x - tau) + ttau) /
                (1 + ttau)).clamp(min=0)
        else:
            raise NotImplementedError()
        self.get_running_cost = get_running_cost
        self.running_cost = None
        self.init_parameters()

    def init_parameters(self, init_gate=0.99):
        if self.gate_activation == 'ReTanH':
            bias_value = 0.5 * math.log((1 + init_gate) / (1 - init_gate))
        elif self.gate_activation == 'Sigmoid':
            bias_value = 0.5 * math.log(init_gate / (1 - init_gate))
        elif self.gate_activation == 'GeReTanH':
            tau = self.gate_activation_kargs['tau']
            bias_value = 0.5 * math.log((1 + init_gate * math.exp(2 * tau)) /
                (1 - init_gate))
        nn.init.normal_(self.gate_conv.weight, std=0.01)
        nn.init.constant_(self.gate_conv.bias, bias_value)

    def encode(self, *inputs):
        outputs = [x.view(x.shape[0] * self.num_groups, -1, *x.shape[2:]) for
            x in inputs]
        return outputs

    def decode(self, *inputs):
        outputs = [x.view(x.shape[0] // self.num_groups, -1, *x.shape[2:]) for
            x in inputs]
        return outputs

    def update_running_cost(self, gate):
        if self.get_running_cost is not None:
            cost = self.get_running_cost(gate)
            if self.running_cost is not None:
                self.running_cost = [(x + y) for x, y in zip(self.
                    running_cost, cost)]
            else:
                self.running_cost = cost

    def clear_running_cost(self):
        self.running_cost = None

    def forward(self, data_input, gate_input, masked_func=None):
        gate = self.gate_activate(self.gate_conv(gate_input))
        self.update_running_cost(gate)
        if masked_func is not None:
            data_input = masked_func(data_input, gate)
        data, gate = self.encode(data_input, gate)
        output, = self.decode(data * gate)
        return output


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 4}]
