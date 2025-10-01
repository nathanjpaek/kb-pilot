import torch
import torch.nn.functional as F
import torch.nn as nn


class GaussionConvF(nn.Module):
    """The first layer in `RobustGCN` that conver node features to distribution (mean, var)"""

    def __init__(self, in_features, out_features, bias=False, gamma=1.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.w = nn.Linear(in_features, out_features, bias=bias)
        self.gamma = gamma

    def reset_parameters(self):
        self.w.reset_parameters()

    def forward(self, x, adj_mean, adj_var):
        h = self.w(x)
        mean = F.elu(h)
        var = F.relu(h)
        attention = torch.exp(-self.gamma * var)
        mean = adj_mean.mm(mean * attention)
        var = adj_var.mm(var * attention * attention)
        return mean, var

    def __repr__(self):
        return (
            f'{self.__class__.__name__}({self.in_features}, {self.out_features})'
            )


def get_inputs():
    return [torch.rand([4, 4]), torch.rand([4, 4]), torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'in_features': 4, 'out_features': 4}]
