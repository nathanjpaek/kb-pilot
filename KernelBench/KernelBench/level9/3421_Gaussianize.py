import torch
import torch.nn as nn


class Gaussianize(nn.Module):
    """ Gaussianization per RealNVP sec 3.6 / fig 4b -- at each step half the variables are directly modeled as Gaussians.
    Model as Gaussians:
        x2 = z2 * exp(logs) + mu, so x2 ~ N(mu, exp(logs)^2) where mu, logs = f(x1)
    then to recover the random numbers z driving the model:
        z2 = (x2 - mu) * exp(-logs)
    Here f(x1) is a conv layer initialized to identity.
    """

    def __init__(self, n_channels):
        super().__init__()
        self.net = nn.Conv2d(n_channels, 2 * n_channels, kernel_size=3,
            padding=1)
        self.log_scale_factor = nn.Parameter(torch.zeros(2 * n_channels, 1, 1))
        self.net.weight.data.zero_()
        self.net.bias.data.zero_()

    def forward(self, x1, x2):
        h = self.net(x1) * self.log_scale_factor.exp()
        m, logs = h[:, 0::2, :, :], h[:, 1::2, :, :]
        z2 = (x2 - m) * torch.exp(-logs)
        logdet = -logs.sum([1, 2, 3])
        return z2, logdet

    def inverse(self, x1, z2):
        h = self.net(x1) * self.log_scale_factor.exp()
        m, logs = h[:, 0::2, :, :], h[:, 1::2, :, :]
        x2 = m + z2 * torch.exp(logs)
        logdet = logs.sum([1, 2, 3])
        return x2, logdet


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'n_channels': 4}]
