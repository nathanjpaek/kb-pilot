import torch
import torch.nn as nn


class IN_self(nn.Module):

    def __init__(self, num_features):
        super(IN_self, self).__init__()
        self.num_features = num_features
        self.gamma = nn.Parameter(torch.Tensor(1, num_features, 1, 1),
            requires_grad=True)
        self.beta = nn.Parameter(torch.Tensor(1, num_features, 1, 1),
            requires_grad=True)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.ones_(self.gamma)
        nn.init.zeros_(self.beta)

    def forward(self, X, eps=1e-05):
        var, mean = torch.var_mean(X, dim=(2, 3), keepdim=True, unbiased=False)
        X = (X - mean) / torch.sqrt(var + eps)
        return self.gamma * X + self.beta


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'num_features': 4}]
