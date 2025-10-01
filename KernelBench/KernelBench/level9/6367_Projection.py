from _paritybench_helpers import _mock_config
import torch
import torch.nn as nn


class TimeDistributed(nn.Module):

    def __init__(self, layer, activation='relu'):
        super().__init__()
        self.layer = layer
        self.activation = self.select_activation(activation)

    def forward(self, x):
        x_reshaped = x.contiguous().view(-1, x.size(-1))
        y = self.layer(x_reshaped)
        y = self.activation(y)
        y = y.contiguous().view(x.size(0), -1, y.size(-1))
        return y

    def select_activation(self, activation):
        if activation == 'relu':
            return nn.ReLU()
        elif activation == 'sigmoid':
            return nn.Sigmoid()
        elif activation == 'tanh':
            return nn.Tanh()
        raise KeyError


class Projection(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.seq_fnn = TimeDistributed(nn.Linear(config['cnn_features'], 3))
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        x = torch.transpose(x, 1, 0)
        x = self.seq_fnn(x)
        x = torch.transpose(x, 1, 0)
        x = self.softmax(x)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'config': _mock_config(cnn_features=4)}]
