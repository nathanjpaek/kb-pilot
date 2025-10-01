import torch
import torch.nn as nn
import torch.optim
import torch.nn.parallel


class LinearDiag(nn.Module):

    def __init__(self, num_features, bias=False):
        super(LinearDiag, self).__init__()
        weight = torch.FloatTensor(num_features).fill_(1)
        self.weight = nn.Parameter(weight, requires_grad=True)
        if bias:
            bias = torch.FloatTensor(num_features).fill_(0)
            self.bias = nn.Parameter(bias, requires_grad=True)
        else:
            self.register_parameter('bias', None)

    def forward(self, X):
        assert X.dim() == 2 and X.size(1) == self.weight.size(0)
        out = X * self.weight.expand_as(X)
        if self.bias is not None:
            out = out + self.bias.expand_as(out)
        return out


def get_inputs():
    return [torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'num_features': 4}]
