import torch
import torch.nn as nn


class Gaussian_transform(nn.Module):

    def __init__(self, output_dim):
        """
        output dim is the number of t parameters in the Gaussian point transformation
        """
        super().__init__()
        self.output_dim = output_dim
        self.t_param = torch.nn.Parameter(torch.randn(output_dim) * 0.1,
            requires_grad=True)
        self.sigma = torch.nn.Parameter(torch.ones(1), requires_grad=True)

    def forward(self, x):
        """
        x is of shape [N,2]
        output is of shape [N,output_dim]
        """
        return torch.exp(-(x[:, :, None] - self.t_param).pow(2).sum(axis=1) /
            (2 * self.sigma.pow(2)))


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'output_dim': 4}]
