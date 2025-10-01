import torch
import torch.nn as nn
from torch.nn.parameter import Parameter


class BatchNormDense(nn.Module):

    def __init__(self, num_features, eps=1e-08):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.gamma = Parameter(torch.Tensor(num_features))
        self.beta = Parameter(torch.Tensor(num_features))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.ones_(self.gamma)
        nn.init.zeros_(self.beta)

    def forward(self, x):
        means = x.mean(dim=0)
        variances = x.var(dim=0)
        x = (x - means) / torch.sqrt(variances + self.eps)
        return self.gamma * x + self.beta


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'num_features': 4}]
