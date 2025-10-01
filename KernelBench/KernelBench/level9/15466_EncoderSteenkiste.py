import torch
from torch import nn


class EncoderSteenkiste(nn.Module):

    def __init__(self, signal_size, latent_dim=10):
        """
        Parameters
        ----------
        signal_size : int for length of signal. Defaults to 30

        latent_dim : int
            Dimensionality of latent output.

        Model Architecture (transposed for decoder)
        ------------
        - 4 convolutional layers (each with 32 channels), (4 x 4 kernel), (stride of 2)
        - 2 fully connected layers (each of 256 units)
        - Latent distribution:
            - 1 fully connected layer of 20 units (log variance and mean for 10 Gaussians)

        References:
            [1] Burgess, Christopher P., et al. "Understanding disentangling in
            $\\beta$-VAE." arXiv preprint arXiv:1804.03599 (2018).
        """
        super(EncoderSteenkiste, self).__init__()
        hidden_dim1 = 50
        hidden_dim2 = 20
        self.latent_dim = latent_dim
        self.img_size = signal_size
        signal_length = signal_size[2]
        self.lin1 = nn.Linear(signal_length, hidden_dim1)
        self.lin2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.lin3 = nn.Linear(hidden_dim2, latent_dim)
        self.mu_logvar_gen = nn.Linear(latent_dim, self.latent_dim * 2)

    def forward(self, x):
        x = torch.relu(self.lin1(x))
        x = torch.relu(self.lin2(x))
        x = torch.relu(self.lin3(x))
        mu_logvar = self.mu_logvar_gen(x)
        mu, logvar = mu_logvar.view(-1, self.latent_dim, 2).unbind(-1)
        return mu, logvar


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'signal_size': [4, 4, 4]}]
