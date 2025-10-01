import torch
from torch import nn


class FeedForward(nn.Module):

    def __init__(self, dhidden, dropout_rate, **kwargs):
        super(FeedForward, self).__init__(**kwargs)
        self.dhidden = dhidden
        self.dropout_rate = dropout_rate
        self.dense_1 = nn.Linear(dhidden, 4 * dhidden)
        self.dense_2 = nn.Linear(4 * dhidden, dhidden)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(dropout_rate)

    def get_config(self):
        config = super(FeedForward, self).get_config()
        config.update({'dhidden': self.dhidden, 'dropout_rate': self.
            dropout_rate})
        return config

    def forward(self, x):
        x = self.dense_1(x)
        x = self.gelu(x)
        x = self.dropout(x)
        x = self.dense_2(x)
        x = self.dropout(x)
        return x


class FNetEncoder(nn.Module):

    def __init__(self, dhidden=512, dropout_rate=0):
        super(FNetEncoder, self).__init__()
        self.feedforward = FeedForward(dhidden, dropout_rate)
        self.LayerNorm_1 = nn.LayerNorm(dhidden)
        self.LayerNorm_2 = nn.LayerNorm(dhidden)

    def get_config(self):
        config = super(FNetEncoder, self).get_config()
        config.update({'dhidden': self.dhidden, 'dropout_rate': self.
            dropout_rate})
        return config

    def forward(self, inputs):
        x_fft = torch.real(torch.fft.fft(torch.fft.fft(inputs).T).T)
        x = self.LayerNorm_1(inputs + x_fft)
        x_ff = self.feedforward(x)
        x = self.LayerNorm_2(x + x_ff)
        return x


def get_inputs():
    return [torch.rand([4, 4, 512, 512])]


def get_init_inputs():
    return [[], {}]
