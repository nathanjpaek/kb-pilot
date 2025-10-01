import torch
import torch.nn as nn


class GlobalpoolFC(nn.Module):

    def __init__(self, num_in, num_class):
        super(GlobalpoolFC, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.fc = nn.Linear(num_in, num_class)

    def forward(self, x):
        y = self.pool(x)
        y = y.reshape(y.shape[0], -1)
        y = self.fc(y)
        return y


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'num_in': 4, 'num_class': 4}]
