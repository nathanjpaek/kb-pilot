import torch
import torch.nn as nn
import torch.nn.functional as F


def weight_init(m):
    if isinstance(m, nn.Linear):
        size = m.weight.size()
        size[0]
        size[1]
        variance = 0.001
        m.weight.data.normal_(0.0, variance)
        try:
            m.bias.data.normal_(0.0, 0.0001)
        except:
            pass


class mlp_layer(nn.Module):

    def __init__(self, input_size, output_size, activation='tanh',
        drouput_prob=0.0):
        super(mlp_layer, self).__init__()
        self.affine = nn.Linear(input_size, output_size)
        weight_init(self.affine)
        if activation.lower() == 'tanh':
            self.activation = torch.tanh
        elif activation.lower() == 'relu':
            self.activation = F.relu()

    def forward(self, x):
        x = self.activation(self.affine(x))
        return x


class merge_layer(nn.Module):

    def __init__(self, hidden_size, output_size, activation='tanh'):
        super(merge_layer, self).__init__()
        self.mlp_layer = mlp_layer(hidden_size, output_size, activation=
            activation)

    def forward(self, x):
        x = torch.mean(x, dim=-1)
        return self.mlp_layer(x)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'hidden_size': 4, 'output_size': 4}]
