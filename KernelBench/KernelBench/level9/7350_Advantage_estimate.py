import torch
import torch.nn as nn
import torch.nn.functional as F


class Advantage_estimate(nn.Module):

    def __init__(self, input_shape, output_shape, device, hidden_shape=128):
        super(Advantage_estimate, self).__init__()
        self.device = device
        self.dropout = nn.Dropout(p=0.01)
        self.input_shape = input_shape
        self.hidden_shape = hidden_shape
        self.output_shape = output_shape
        self.l1 = nn.Linear(self.input_shape, self.hidden_shape)
        self.l2 = nn.Linear(self.hidden_shape, self.output_shape)

    def forward(self, x):
        x = x
        x = self.dropout(F.leaky_relu(self.l1(x)))
        x = self.l2(x)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'input_shape': 4, 'output_shape': 4, 'device': 0}]
