import math
import torch
from torch import nn
from torch.nn.parameter import Parameter


class LinearCapsPro(nn.Module):

    def __init__(self, in_features, num_C, num_D, eps=0.0001):
        super(LinearCapsPro, self).__init__()
        self.in_features = in_features
        self.num_C = num_C
        self.num_D = num_D
        self.eps = eps
        self.weight = Parameter(torch.Tensor(num_C * num_D, in_features))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, x, eye):
        weight_caps = self.weight[:self.num_D]
        sigma = torch.inverse(torch.mm(weight_caps, torch.t(weight_caps)) +
            self.eps * eye)
        sigma = torch.unsqueeze(sigma, dim=0)
        for ii in range(1, self.num_C):
            weight_caps = self.weight[ii * self.num_D:(ii + 1) * self.num_D]
            sigma_ = torch.inverse(torch.mm(weight_caps, torch.t(
                weight_caps)) + self.eps * eye)
            sigma_ = torch.unsqueeze(sigma_, dim=0)
            sigma = torch.cat((sigma, sigma_))
        out = torch.matmul(x, torch.t(self.weight))
        out = out.view(out.shape[0], self.num_C, 1, self.num_D)
        out = torch.matmul(out, sigma)
        out = torch.matmul(out, self.weight.view(self.num_C, self.num_D,
            self.in_features))
        out = torch.squeeze(out, dim=2)
        out = torch.matmul(out, torch.unsqueeze(x, dim=2))
        out = torch.squeeze(out, dim=2)
        return torch.sqrt(out)


def get_inputs():
    return [torch.rand([4, 4]), torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'in_features': 4, 'num_C': 4, 'num_D': 4}]
