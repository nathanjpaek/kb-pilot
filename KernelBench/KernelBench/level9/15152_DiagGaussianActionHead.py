import torch
import numpy as np
import torch.optim
import torch.nn as nn
import torch.nn.init as init
import torch.nn.utils
import torch.autograd


class DiagGaussianActionHead(nn.Module):
    """
    Action head where actions are normally distibuted uncorrelated variables with specific means and variances.

    Means are calculated directly from the network while standard deviation are a parameter of this module
    """
    LOG2PI = np.log(2.0 * np.pi)

    def __init__(self, input_dim, num_dimensions):
        super().__init__()
        self.input_dim = input_dim
        self.num_dimensions = num_dimensions
        self.linear_layer = nn.Linear(input_dim, num_dimensions)
        self.log_std = nn.Parameter(torch.zeros(1, num_dimensions))

    def forward(self, input_data):
        means = self.linear_layer(input_data)
        log_std_tile = self.log_std.repeat(means.size(0), 1)
        return torch.stack([means, log_std_tile], dim=-1)

    def sample(self, params, argmax_sampling=False):
        """ Sample from a probability space of all actions """
        means = params[:, :, 0]
        log_std = params[:, :, 1]
        if argmax_sampling:
            return means
        else:
            return torch.randn_like(means) * torch.exp(log_std) + means

    def logprob(self, action_sample, pd_params):
        """ Log-likelihood """
        means = pd_params[:, :, 0]
        log_std = pd_params[:, :, 1]
        std = torch.exp(log_std)
        z_score = (action_sample - means) / std
        return -(0.5 * (z_score ** 2 + self.LOG2PI).sum(dim=-1) + log_std.
            sum(dim=-1))

    def reset_weights(self):
        init.orthogonal_(self.linear_layer.weight, gain=0.01)
        init.constant_(self.linear_layer.bias, 0.0)

    def entropy(self, params):
        """
        Categorical distribution entropy calculation - sum probs * log(probs).
        In case of diagonal gaussian distribution - 1/2 log(2 pi e sigma^2)
        """
        log_std = params[:, :, 1]
        return (log_std + 0.5 * (self.LOG2PI + 1)).sum(dim=-1)

    def kl_divergence(self, params_q, params_p):
        """
        Categorical distribution KL divergence calculation
        KL(Q || P) = sum Q_i log (Q_i / P_i)

        Formula is:
        log(sigma_p) - log(sigma_q) + (sigma_q^2 + (mu_q - mu_p)^2))/(2 * sigma_p^2)
        """
        means_q = params_q[:, :, 0]
        log_std_q = params_q[:, :, 1]
        means_p = params_p[:, :, 0]
        log_std_p = params_p[:, :, 1]
        std_q = torch.exp(log_std_q)
        std_p = torch.exp(log_std_p)
        kl_div = log_std_p - log_std_q + (std_q ** 2 + (means_q - means_p) ** 2
            ) / (2.0 * std_p ** 2) - 0.5
        return kl_div.sum(dim=-1)


def get_inputs():
    return [torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'input_dim': 4, 'num_dimensions': 4}]
