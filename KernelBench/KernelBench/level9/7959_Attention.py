import torch
import torch.nn as nn


class Attention(nn.Module):

    def __init__(self, input_dim, feature_dim):
        super(Attention, self).__init__()
        self.feature_dim = feature_dim
        self.input_dim = input_dim
        weight = torch.zeros(self.feature_dim, self.feature_dim)
        nn.init.kaiming_uniform_(weight)
        self.weight = nn.Parameter(weight)
        w = torch.zeros(1, self.feature_dim)
        nn.init.kaiming_uniform_(w)
        self.w = nn.Parameter(w)

    def forward(self, input, context=None):
        u = torch.matmul(input.contiguous().view(-1, self.feature_dim),
            self.weight).view(-1, self.feature_dim)
        u = torch.tanh(torch.matmul(self.w, u.view(self.feature_dim, -1)))
        if context is not None:
            u = u * context
        a = torch.exp(u)
        a = a / (torch.sum(a, 1, keepdim=True) + 1e-10)
        weighted_input = torch.matmul(input.view(self.feature_dim, -1), a.
            view(-1, 1))
        return torch.sum(weighted_input, 1)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'input_dim': 4, 'feature_dim': 4}]
