import torch
import torch.nn as nn
import torch.nn.functional as F


class GaussionConvD(nn.Module):
    """The subsequent layer in `RobustGCN` that takes node distribution (mean, var) as input"""

    def __init__(self, in_features, out_features, bias=False, gamma=1.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.w_mean = nn.Linear(in_features, out_features, bias=bias)
        self.w_var = nn.Linear(in_features, out_features, bias=bias)
        self.gamma = gamma

    def reset_parameters(self):
        self.w.reset_parameters()

    def forward(self, mean, var, adj_mean, adj_var):
        mean = F.elu(self.w_mean(mean))
        var = F.relu(self.w_var(var))
        attention = torch.exp(-self.gamma * var)
        mean = adj_mean.mm(mean * attention)
        var = adj_var.mm(var * attention * attention)
        return mean, var

    def __repr__(self):
        return (
            f'{self.__class__.__name__}({self.in_features}, {self.out_features})'
            )


def get_inputs():
    return [torch.rand([4, 4]), torch.rand([4, 4]), torch.rand([4, 4]),
        torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'in_features': 4, 'out_features': 4}]
