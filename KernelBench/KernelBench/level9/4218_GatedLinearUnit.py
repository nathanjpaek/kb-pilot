import torch
import torch.nn as nn


class GatedLinearUnit(nn.Module):

    def __init__(self, input_size, hidden_layer_size, dropout_rate,
        activation=None):
        super(GatedLinearUnit, self).__init__()
        self.input_size = input_size
        self.hidden_layer_size = hidden_layer_size
        self.dropout_rate = dropout_rate
        self.activation_name = activation
        if self.dropout_rate:
            self.dropout = nn.Dropout(p=self.dropout_rate)
        self.W4 = torch.nn.Linear(self.input_size, self.hidden_layer_size)
        self.W5 = torch.nn.Linear(self.input_size, self.hidden_layer_size)
        if self.activation_name:
            self.activation = getattr(nn, self.activation_name)()
        self.sigmoid = nn.Sigmoid()
        self.init_weights()

    def init_weights(self):
        for n, p in self.named_parameters():
            if 'bias' not in n:
                torch.nn.init.xavier_uniform_(p)
            elif 'bias' in n:
                torch.nn.init.zeros_(p)

    def forward(self, x):
        if self.dropout_rate:
            x = self.dropout(x)
        if self.activation_name:
            output = self.sigmoid(self.W4(x)) * self.activation(self.W5(x))
        else:
            output = self.sigmoid(self.W4(x)) * self.W5(x)
        return output


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'input_size': 4, 'hidden_layer_size': 1, 'dropout_rate': 0.5}]
