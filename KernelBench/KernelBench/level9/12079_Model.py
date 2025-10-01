import torch
from torch import nn


class Model(nn.Module):

    def __init__(self, input_size, dropout=0.5):
        super(Model, self).__init__()
        self.dropout = dropout
        if self.dropout > 0:
            self.dropout = nn.Dropout(dropout)
        self.encode_w1 = nn.Linear(input_size, 64)
        self.encode_w2 = nn.Linear(64, 32)
        self.decode_w1 = nn.Linear(32, 64)
        self.decode_w2 = nn.Linear(64, input_size)

    def encoder(self, x):
        x = self.encode_w1(x)
        x = torch.relu(x)
        x = self.encode_w2(x)
        x = torch.relu(x)
        if self.dropout:
            x = self.dropout(x)
        return x

    def decoder(self, x):
        x = self.decode_w1(x)
        x = torch.relu(x)
        x = self.decode_w2(x)
        return x

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'input_size': 4}]
