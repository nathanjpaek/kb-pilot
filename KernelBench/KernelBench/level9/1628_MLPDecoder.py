import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPDecoder(nn.Module):

    def __init__(self, d_in, d_out):
        super(MLPDecoder, self).__init__()
        H1 = 10
        H2 = 100
        self._d_in = d_in
        self._d_out = d_out
        self.l1 = nn.Linear(d_in, H1)
        self.l11 = nn.Linear(H1, H2)
        self.l2 = nn.Linear(H2, d_out)

    def forward(self, input):
        x = self.l1(input)
        x = F.relu(x)
        x = self.l11(x)
        x = F.relu(x)
        x = self.l2(x)
        return x

    def input_size(self):
        return self._d_in

    def output_size(self):
        return self._d_out


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'d_in': 4, 'd_out': 4}]
