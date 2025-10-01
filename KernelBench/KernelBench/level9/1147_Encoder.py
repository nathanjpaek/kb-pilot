import torch
import torch.nn as nn


class Encoder(nn.Module):

    def __init__(self):
        super(Encoder, self).__init__()
        self.Conv1 = nn.Conv2d(1, 16, 3, 1, 1)
        self.Relu = nn.ReLU(inplace=True)
        self.layers = nn.ModuleDict({'DenseConv1': nn.Conv2d(16, 16, 3, 1, 
            1), 'DenseConv2': nn.Conv2d(32, 16, 3, 1, 1), 'DenseConv3': nn.
            Conv2d(48, 16, 3, 1, 1)})

    def forward(self, x):
        x = self.Relu(self.Conv1(x))
        for i in range(len(self.layers)):
            out = self.layers['DenseConv' + str(i + 1)](x)
            x = torch.cat([x, out], 1)
        return x


def get_inputs():
    return [torch.rand([4, 1, 64, 64])]


def get_init_inputs():
    return [[], {}]
