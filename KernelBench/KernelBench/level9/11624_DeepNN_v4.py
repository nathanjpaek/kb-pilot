import torch
import torch.nn as nn


class DeepNN_v4(nn.Module):

    def __init__(self, X_dim, i_dropout_rate, h_dropout_rate):
        super().__init__()
        self.v4_layer1 = nn.Linear(X_dim, 128, bias=True)
        self.v4_layer2 = nn.Linear(128, 128, bias=True)
        self.v4_layer3 = nn.Linear(128, 64, bias=True)
        self.v4_layer4 = nn.Linear(64, 1, bias=True)
        self.i_dropout = nn.Dropout(i_dropout_rate)
        self.h_dropout = nn.Dropout(h_dropout_rate)

    def forward(self, x):
        x = self.i_dropout(torch.tanh(self.v4_layer1(x)))
        x = self.h_dropout(torch.relu(self.v4_layer2(x)))
        x = self.h_dropout(torch.relu(self.v4_layer3(x)))
        x = torch.sigmoid(self.v4_layer4(x))
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'X_dim': 4, 'i_dropout_rate': 0.5, 'h_dropout_rate': 0.5}]
