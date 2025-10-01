import torch
import torch.nn as nn


class SelfAttention(nn.Module):

    def __init__(self, input_dim):
        super(SelfAttention, self).__init__()
        self.pre_pooling_linear = nn.Linear(input_dim, input_dim)
        self.pooling_linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        self.pre_pooling_linear(x)
        weights = self.pooling_linear(torch.tanh(self.pre_pooling_linear(x))
            ).squeeze(dim=2)
        weights = nn.Softmax(dim=-1)(weights)
        return torch.mul(x, weights.unsqueeze(2)).sum(dim=1)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'input_dim': 4}]
