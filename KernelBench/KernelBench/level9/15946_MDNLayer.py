import torch
from torch import nn
from torch.nn import functional as F


class MDNLayer(nn.Module):
    """ Mixture Density Network layer

    The input maps to the parameters of a Mixture of Gaussians (MoG) probability
    distribution, where each Gaussian has out_dim dimensions and diagonal covariance.
    If dim_wise is True, features for each dimension are modeld by independent 1-D GMMs
    instead of modeling jointly. This would workaround training difficulty
    especially for high dimensional data.

    Implementation references:
    1. Mixture Density Networks by Mike Dusenberry https://mikedusenberry.com/mixture-density-networks
    2. PRML book https://www.microsoft.com/en-us/research/people/cmbishop/prml-book/
    3. sagelywizard/pytorch-mdn https://github.com/sagelywizard/pytorch-mdn
    4. sksq96/pytorch-mdn https://github.com/sksq96/pytorch-mdn

    Attributes:
        in_dim (int): the number of dimensions in the input
        out_dim (int): the number of dimensions in the output
        num_gaussians (int): the number of mixture component
        dim_wise (bool): whether to model data for each dimension seperately
    """

    def __init__(self, in_dim, out_dim, num_gaussians=30, dim_wise=False):
        super(MDNLayer, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_gaussians = num_gaussians
        self.dim_wise = dim_wise
        odim_log_pi = out_dim * num_gaussians if dim_wise else num_gaussians
        self.log_pi = nn.Linear(in_dim, odim_log_pi)
        self.log_sigma = nn.Linear(in_dim, out_dim * num_gaussians)
        self.mu = nn.Linear(in_dim, out_dim * num_gaussians)

    def forward(self, minibatch):
        """Forward for MDN

        Args:
            minibatch (torch.Tensor): tensor of shape (B, T, D_in)
                B is the batch size and T is data lengths of this batch,
                and D_in is in_dim.

        Returns:
            torch.Tensor: Tensor of shape (B, T, G) or (B, T, G, D_out)
                Log of mixture weights. G is num_gaussians and D_out is out_dim.
            torch.Tensor: Tensor of shape (B, T, G, D_out)
                the log of standard deviation of each Gaussians.
            torch.Tensor: Tensor of shape (B, T, G, D_out)
                mean of each Gaussians
        """
        B = len(minibatch)
        if self.dim_wise:
            log_pi = self.log_pi(minibatch).view(B, -1, self.num_gaussians,
                self.out_dim)
            log_pi = F.log_softmax(log_pi, dim=2)
        else:
            log_pi = F.log_softmax(self.log_pi(minibatch), dim=2)
        log_sigma = self.log_sigma(minibatch)
        log_sigma = log_sigma.view(B, -1, self.num_gaussians, self.out_dim)
        mu = self.mu(minibatch)
        mu = mu.view(B, -1, self.num_gaussians, self.out_dim)
        return log_pi, log_sigma, mu


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_dim': 4, 'out_dim': 4}]
