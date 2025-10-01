import torch
import torch.nn as nn


class Triangle_transform(nn.Module):

    def __init__(self, output_dim):
        """
        output dim is the number of t parameters in the triangle point transformation
        """
        super().__init__()
        self.output_dim = output_dim
        self.t_param = torch.nn.Parameter(torch.randn(output_dim) * 0.1,
            requires_grad=True)

    def forward(self, x):
        """
        x is of shape [N,2]
        output is of shape [N,output_dim]
        """
        return torch.nn.functional.relu(x[:, 1][:, None] - torch.abs(self.
            t_param - x[:, 0][:, None]))


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'output_dim': 4}]
