import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):

    def __init__(self, _input_size: 'int', _output_size: 'int',
        _hidden_layers: 'int', _hidden_size: 'int'):
        super(Net, self).__init__()
        self.input = nn.Linear(_input_size, _hidden_size)
        self.hidden_layers = _hidden_layers
        self.hidden = []
        for i in range(_hidden_layers):
            layer = nn.Linear(_hidden_size, _hidden_size)
            self.add_module('h' + str(i), layer)
            self.hidden.append(layer)
        self.advantage = nn.Linear(_hidden_size, _output_size)
        self.value = nn.Linear(_hidden_size, 1)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0.01)

    def forward(self, x):
        x = F.relu(self.input(x))
        for i in range(self.hidden_layers):
            x = F.relu(self.hidden[i](x))
        value = self.value(x)
        raw_advantage = self.advantage(x)
        advantage = raw_advantage - raw_advantage.mean(dim=-1, keepdim=True)
        q_value = value + advantage
        return q_value


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'_input_size': 4, '_output_size': 4, '_hidden_layers': 1,
        '_hidden_size': 4}]
