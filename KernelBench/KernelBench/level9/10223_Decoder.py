import torch
import torch.nn.functional as F
from torch import nn


def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)


class Decoder(torch.nn.Module):

    def __init__(self, input_dim, out_dim, hidden_size=128):
        super(Decoder, self).__init__()
        self.linear1 = torch.nn.Linear(input_dim, hidden_size)
        self.linear2 = torch.nn.Linear(hidden_size, hidden_size)
        self.linear3 = torch.nn.Linear(hidden_size, out_dim)
        self.apply(weights_init_)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'input_dim': 4, 'out_dim': 4}]
