import torch
import numpy as np
from torch import tensor
import torch.nn as nn
import numpy.random as rng


class BaseFlow(nn.Module):
    """ """

    def __init__(self, n_inputs, **kwargs):
        super(BaseFlow, self).__init__()
        self.n_inputs = n_inputs

    def forward(self, x, **kwargs):
        raise NotImplementedError

    def generate_samples(self, n_samples=1, u=None, **kwargs):
        raise NotImplementedError

    def log_likelihood(self, x, **kwargs):
        """ Calculates log p(x) with a Gaussian base density """
        u, logdet_dudx = self.forward(x, **kwargs)
        constant = float(-0.5 * self.n_inputs * np.log(2.0 * np.pi))
        log_likelihood = constant - 0.5 * torch.sum(u ** 2, dim=1
            ) + logdet_dudx
        return u, log_likelihood

    def log_likelihood_and_score(self, x, **kwargs):
        """ Calculates log p(x) and t(x) with a Gaussian base density """
        u, log_likelihood = self.log_likelihood(x, **kwargs)
        return u, log_likelihood, None


class BatchNorm(BaseFlow):
    """BatchNorm implementation"""

    def __init__(self, n_inputs, alpha=0.1, eps=1e-05):
        super(BatchNorm, self).__init__(n_inputs)
        self.n_inputs = n_inputs
        self.alpha = alpha
        self.eps = eps
        self.calculated_running_mean = False
        self.running_mean = torch.zeros(self.n_inputs)
        self.running_var = torch.zeros(self.n_inputs)

    def forward(self, x, fixed_params=False):
        """Calculates x -> u(x) (batch norming)"""
        if fixed_params:
            mean = self.running_mean
            var = self.running_var
        else:
            mean = torch.mean(x, dim=0)
            var = torch.mean((x - mean) ** 2, dim=0) + self.eps
            if not self.calculated_running_mean:
                self.running_mean = mean
                self.running_var = var
            else:
                self.running_mean = (1.0 - self.alpha
                    ) * self.running_mean + self.alpha * mean
                self.running_var = (1.0 - self.alpha
                    ) * self.running_var + self.alpha * var
                self.calculated_running_mean = True
        u = (x - mean) / torch.sqrt(var)
        logdet = -0.5 * torch.sum(torch.log(var))
        return u, logdet

    def inverse(self, u):
        """Calculates u -> x(u) (the approximate inverse transformation based on running mean and variance)"""
        x = torch.sqrt(self.running_var) * u + self.running_mean
        return x

    def generate_samples(self, n_samples=1, u=None, **kwargs):
        if u is None:
            u = tensor(rng.randn(n_samples, self.n_inputs))
        x = torch.sqrt(self.running_var) * u + self.running_mean
        return x

    def to(self, *args, **kwargs):
        logger.debug('Transforming BatchNorm to %s', args)
        self = super()
        self.running_mean = self.running_mean
        self.running_var = self.running_var
        return self


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'n_inputs': 4}]
