import torch
import torch.nn as nn


class DiagGaussianActor(nn.Module):
    """torch.distributions implementation of an diagonal Gaussian policy."""

    def __init__(self, log_std_bounds=[-5, 2]):
        super().__init__()
        self.log_std_bounds = log_std_bounds

    def forward(self, mu, log_std):
        log_std = torch.tanh(log_std)
        log_std_min, log_std_max = self.log_std_bounds
        log_std = log_std_min + 0.5 * (log_std_max - log_std_min) * (log_std +
            1)
        std = log_std.exp()
        mu = torch.tanh(mu)
        return mu, std


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
