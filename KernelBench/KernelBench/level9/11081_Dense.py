import torch
import torch.nn as nn
import torch.utils.data


class Dense(nn.Module):

    def __init__(self, num_channels, num_filters, filter_size, dropout_prob):
        super().__init__()
        self.dense_conv = nn.Conv2d(in_channels=num_channels, out_channels=
            num_filters, kernel_size=filter_size, stride=1, padding=0, bias
            =False)
        self.hardswish = nn.Hardswish()
        self.dropout = nn.Dropout(p=dropout_prob)

    def forward(self, x):
        x = self.dense_conv(x)
        x = self.hardswish(x)
        x = self.dropout(x)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'num_channels': 4, 'num_filters': 4, 'filter_size': 4,
        'dropout_prob': 0.5}]
