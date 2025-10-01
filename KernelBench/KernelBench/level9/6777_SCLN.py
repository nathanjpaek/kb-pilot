import torch
import torch.nn as nn


class LinearNorm(nn.Module):
    """ LinearNorm Projection """

    def __init__(self, in_features, out_features, bias=False):
        super(LinearNorm, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias)
        nn.init.xavier_uniform_(self.linear.weight)
        if bias:
            nn.init.constant_(self.linear.bias, 0.0)

    def forward(self, x):
        x = self.linear(x)
        return x


class SCLN(nn.Module):
    """ Speaker Condition Layer Normalization """

    def __init__(self, s_size, hidden_size, eps=1e-08, bias=False):
        super(SCLN, self).__init__()
        self.hidden_size = hidden_size
        self.affine_layer = LinearNorm(s_size, 2 * hidden_size, bias)
        self.eps = eps

    def forward(self, x, s):
        mu, sigma = torch.mean(x, dim=-1, keepdim=True), torch.std(x, dim=-
            1, keepdim=True)
        y = (x - mu) / (sigma + self.eps)
        b, g = torch.split(self.affine_layer(s), self.hidden_size, dim=-1)
        o = g * y + b
        return o


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'s_size': 4, 'hidden_size': 4}]
