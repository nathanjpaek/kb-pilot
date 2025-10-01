import torch
import torch.nn as nn


class DeepNN_v1(nn.Module):

    def __init__(self, X_dim, i_dropout_rate, h_dropout_rate):
        super().__init__()
        self.v1_layer1 = nn.Linear(X_dim, 512, bias=True)
        self.v1_layer2 = nn.Linear(512, 1, bias=True)
        self.i_dropout = nn.Dropout(i_dropout_rate)
        self.h_dropout = nn.Dropout(h_dropout_rate)

    def forward(self, x):
        x = self.i_dropout(torch.tanh(self.v1_layer1(x)))
        x = torch.sigmoid(self.v1_layer2(x))
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'X_dim': 4, 'i_dropout_rate': 0.5, 'h_dropout_rate': 0.5}]
