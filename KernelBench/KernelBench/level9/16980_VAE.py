import torch
import numpy as np
import torch.utils.data
import torch.nn as nn


class VAE(nn.Module):

    def __init__(self, input_size, latent_size):
        super(VAE, self).__init__()
        self.latent_size = latent_size
        self.input_size = input_size
        self.mu_layer = nn.Linear(self.input_size, self.latent_size)
        self.logvar_layer = nn.Linear(self.input_size, self.latent_size)

    def _reparametrize(self, mu, logvar):
        std_dev = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std_dev)
        return eps.mul(std_dev).add_(mu)

    def forward(self, input):
        mu = self.mu_layer(input)
        logvar = self.logvar_layer(input)
        z = self._reparametrize(mu, logvar)
        return z, mu, logvar

    def default_loss(self, x, recon_x, mu, logvar):
        BCE = nn.functional.cross_entropy(recon_x, x)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return BCE + KLD

    def kl_anneal_function(self, step, k=0.0025, x0=2500, anneal_function=
        'linear'):
        if anneal_function == 'logistic':
            return float(1 / (1 + np.exp(-k * (step - x0))))
        elif anneal_function == 'linear':
            return min(1, step / x0)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'input_size': 4, 'latent_size': 4}]
