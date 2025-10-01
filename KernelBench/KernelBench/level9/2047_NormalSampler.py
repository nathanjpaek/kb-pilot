import torch
from torch import nn


class NormalSampler(nn.Module):
    """p(z)"""

    def __init__(self):
        super(NormalSampler, self).__init__()
        self.register_buffer('eps', torch.tensor(1e-10))

    def forward(self, mean, log_var):
        epsilon = torch.randn(mean.size(), requires_grad=False, device=mean
            .device)
        std = log_var.mul(0.5).exp_()
        z = mean.addcmul(std, epsilon)
        return z

    def kl_divergence(self, mean, log_var, z):
        """
        L elbo(x) = Eq(z|x)[log p(x|z)] - KL(q(z|x)||p(z))
        D_{KL}(q(z|x)||p(z))
        """
        return -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp(),
            dim=1)


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
