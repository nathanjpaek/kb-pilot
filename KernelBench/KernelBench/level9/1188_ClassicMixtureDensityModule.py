import torch
from torch import nn


class ClassicMixtureDensityModule(nn.Module):

    def __init__(self, dim_input, dim_output, num_components):
        super(ClassicMixtureDensityModule, self).__init__()
        self.dim_input = dim_input
        self.dim_output = dim_output
        self.M = num_components
        self.layer_mapping = nn.Linear(dim_input, (2 * dim_output + 1) *
            num_components)
        self.layer_alpha = nn.Softmax(dim=1)

    def forward(self, x):
        p = self.layer_mapping(x)
        alpha = self.layer_alpha(p[:, :self.M])
        mu = p[:, self.M:(self.dim_output + 1) * self.M]
        sigma = torch.exp(p[:, (self.dim_output + 1) * self.M:])
        mu = mu.view(-1, self.M, self.dim_output)
        sigma = sigma.view(-1, self.M, self.dim_output)
        return alpha, mu, sigma


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'dim_input': 4, 'dim_output': 4, 'num_components': 4}]
