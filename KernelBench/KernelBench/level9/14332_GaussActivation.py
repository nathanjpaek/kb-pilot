import torch
import torch.nn as nn
from torch.nn.parameter import Parameter


class GaussActivation(nn.Module):

    def __init__(self, a, mu, sigma1, sigma2):
        super(GaussActivation, self).__init__()
        self.a = Parameter(torch.tensor(a, dtype=torch.float32))
        self.mu = Parameter(torch.tensor(mu, dtype=torch.float32))
        self.sigma1 = Parameter(torch.tensor(sigma1, dtype=torch.float32))
        self.sigma2 = Parameter(torch.tensor(sigma2, dtype=torch.float32))

    def forward(self, inputFeatures):
        self.a.data = torch.clamp(self.a.data, 1.01, 6.0)
        self.mu.data = torch.clamp(self.mu.data, 0.1, 3.0)
        self.sigma1.data = torch.clamp(self.sigma1.data, 0.5, 2.0)
        self.sigma2.data = torch.clamp(self.sigma2.data, 0.5, 2.0)
        lowerThanMu = inputFeatures < self.mu
        largerThanMu = inputFeatures >= self.mu
        leftValuesActiv = self.a * torch.exp(-self.sigma1 * (inputFeatures -
            self.mu) ** 2)
        leftValuesActiv.masked_fill_(largerThanMu, 0.0)
        rightValueActiv = 1 + (self.a - 1) * torch.exp(-self.sigma2 * (
            inputFeatures - self.mu) ** 2)
        rightValueActiv.masked_fill_(lowerThanMu, 0.0)
        output = leftValuesActiv + rightValueActiv
        return output


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'a': 4, 'mu': 4, 'sigma1': 4, 'sigma2': 4}]
