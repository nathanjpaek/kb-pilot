import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter


class Relu_Adpt(nn.Module):

    def __init__(self, num_C, num_D, eps=0.0001):
        super(Relu_Adpt, self).__init__()
        self.num_C = num_C
        self.num_D = num_D
        self.eps = eps
        self.theta = Parameter(torch.Tensor(1, self.num_C, 1, 1, 1))
        self.theta.data.fill_(0.0)

    def forward(self, x):
        x_caps = x.view(x.shape[0], self.num_C, self.num_D, x.shape[2], x.
            shape[3])
        x_length = torch.sqrt(torch.sum(x_caps * x_caps, dim=2))
        x_length = torch.unsqueeze(x_length, 2)
        x_caps = F.relu(x_length - self.theta) * x_caps / (x_length + self.eps)
        x = x_caps.view(x.shape[0], -1, x.shape[2], x.shape[3])
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'num_C': 4, 'num_D': 4}]
