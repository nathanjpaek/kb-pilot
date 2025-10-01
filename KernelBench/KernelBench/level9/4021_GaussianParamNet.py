import torch
import torch.nn as nn
import torch.nn.functional as F


class GaussianParamNet(nn.Module):
    """
    Parameterise a Gaussian distributions.
    """

    def __init__(self, input_dim, output_dim):
        super(GaussianParamNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, input_dim, bias=False)
        self.layer_nml = nn.LayerNorm(input_dim, elementwise_affine=False)
        self.fc2 = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        """
        x: input image with shape [B, K, 2*D]
        """
        x = self.fc2(F.relu(self.layer_nml(self.fc1(x))))
        mu, sigma = x.chunk(2, dim=-1)
        sigma = F.softplus(sigma + 0.5) + 1e-08
        return mu, sigma


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'input_dim': 4, 'output_dim': 4}]
