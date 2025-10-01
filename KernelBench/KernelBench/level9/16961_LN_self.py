import torch
import torch.nn as nn


class LN_self(nn.Module):

    def __init__(self, num_features):
        super().__init__()
        shape = 1, num_features, 1, 1
        self.gamma = nn.Parameter(torch.ones(shape))
        self.beta = nn.Parameter(torch.zeros(shape))

    def forward(self, X, eps=1e-05):
        var, mean = torch.var_mean(X, dim=(1, 2, 3), keepdim=True, unbiased
            =False)
        X = (X - mean) / torch.sqrt(var + eps)
        return self.gamma * X + self.beta


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'num_features': 4}]
