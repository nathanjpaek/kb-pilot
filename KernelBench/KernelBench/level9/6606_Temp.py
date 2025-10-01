import torch
import torch.nn as nn
import torch.nn.functional as F


class Temp(nn.Module):

    def __init__(self, input_dim, output_dim):
        super(Temp, self).__init__()
        self.linear1 = nn.Linear(input_dim, 256)
        self.linear2 = nn.Linear(256, 256)
        self.linear3 = nn.Linear(256, 256)
        self.linear4 = nn.Linear(256, output_dim)

    def forward(self, x):
        x = F.leaky_relu(self.linear1(x))
        x = F.sigmoid(self.linear2(x))
        x = F.leaky_relu(self.linear3(x))
        x = self.linear4(x)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'input_dim': 4, 'output_dim': 4}]
