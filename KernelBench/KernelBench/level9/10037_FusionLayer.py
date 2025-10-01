import torch
import torch.nn as nn


class FusionLayer(nn.Module):
    """
    vector based fusion
    m(x, y) = W([x, y, x * y, x - y]) + b
    g(x, y) = w([x, y, x * y, x - y]) + b
    :returns g(x, y) * m(x, y) + (1 - g(x, y)) * x
    """

    def __init__(self, input_dim):
        super(FusionLayer, self).__init__()
        self.linear_f = nn.Linear(input_dim * 4, input_dim, bias=True)
        self.linear_g = nn.Linear(input_dim * 4, 1, bias=True)
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, y):
        z = torch.cat([x, y, x * y, x - y], dim=2)
        gated = self.sigmoid(self.linear_g(z))
        fusion = self.tanh(self.linear_f(z))
        return gated * fusion + (1 - gated) * x


def get_inputs():
    return [torch.rand([4, 4, 4]), torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'input_dim': 4}]
