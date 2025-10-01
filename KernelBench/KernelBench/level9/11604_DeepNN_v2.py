import torch
import torch.nn as nn


class DeepNN_v2(nn.Module):

    def __init__(self, X_dim, i_dropout_rate, h_dropout_rate):
        super().__init__()
        self.v2_layer1 = nn.Linear(X_dim, 256, bias=True)
        self.v2_layer2 = nn.Linear(256, 256, bias=True)
        self.v2_layer3 = nn.Linear(256, 1, bias=True)
        self.i_dropout = nn.Dropout(i_dropout_rate)
        self.h_dropout = nn.Dropout(h_dropout_rate)

    def forward(self, x):
        x = self.i_dropout(torch.tanh(self.v2_layer1(x)))
        x = self.h_dropout(torch.tanh(self.v2_layer2(x)))
        x = torch.sigmoid(self.v2_layer3(x))
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'X_dim': 4, 'i_dropout_rate': 0.5, 'h_dropout_rate': 0.5}]
