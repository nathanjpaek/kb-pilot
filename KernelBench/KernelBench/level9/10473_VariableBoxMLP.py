import torch
import torch.optim
import torch.jit
import torch.nn as nn


class VariableBoxMLP(nn.Module):

    def __init__(self, num_in_features: 'int', num_out_features: 'int',
        neurons_per_layer: 'int', hidden_layers: 'int'):
        super(VariableBoxMLP, self).__init__()
        self.hidden_layers = hidden_layers
        self.act = nn.ELU()
        self.l_in = nn.Linear(in_features=num_in_features, out_features=
            neurons_per_layer)
        for i in range(0, hidden_layers):
            layer = nn.Linear(in_features=neurons_per_layer, out_features=
                neurons_per_layer)
            torch.nn.init.xavier_normal_(layer.weight)
            torch.nn.init.zeros_(layer.bias)
            setattr(self, 'l' + str(i), layer)
        self.l_out = nn.Linear(in_features=neurons_per_layer, out_features=
            num_out_features)
        torch.nn.init.xavier_normal_(self.l_in.weight)
        torch.nn.init.zeros_(self.l_in.bias)
        torch.nn.init.xavier_normal_(self.l_out.weight)
        torch.nn.init.zeros_(self.l_out.bias)

    def forward(self, x):
        x = self.act(self.l_in(x))
        for i in range(self.hidden_layers):
            x = self.act(getattr(self, 'l' + str(i)).__call__(x))
        x = self.l_out(x)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'num_in_features': 4, 'num_out_features': 4,
        'neurons_per_layer': 1, 'hidden_layers': 1}]
