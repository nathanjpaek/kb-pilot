import math
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter


class BatchDense(nn.Module):

    def __init__(self, batch, in_features, out_features, bias_init=None):
        super(BatchDense, self).__init__()
        self.batch = batch
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(batch, in_features, out_features))
        self.bias = Parameter(torch.Tensor(batch, 1, out_features))
        self.reset_parameters(bias_init)

    def reset_parameters(self, bias_init=None):
        stdv = math.sqrt(6.0 / (self.in_features + self.out_features))
        self.weight.data.uniform_(-stdv, stdv)
        if bias_init is not None:
            self.bias.data = torch.from_numpy(bias_init)
        else:
            self.bias.data.fill_(0)

    def forward(self, x):
        x.size()
        x = x.view(x.size(0), self.batch, -1)
        out = x.transpose(0, 1).contiguous()
        out = torch.baddbmm(self.bias, out, self.weight)
        out = out.transpose(0, 1).contiguous()
        out = out.view(x.size(0), -1)
        return out


def get_inputs():
    return [torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'batch': 4, 'in_features': 4, 'out_features': 4}]
