import torch
import torch.nn.functional as F
from torch import nn


class VAE(nn.Module):
    """A classic VAE.

    Params
    ------
    input_dim : int
        The size of the (flattened) image vector 
    latent_dim : int
        The size of the latent memory    
    """

    def __init__(self, input_dim=784, latent_dim=20):
        super(VAE, self).__init__()
        self.input_dim = int(input_dim)
        self.latent_dim = int(latent_dim)
        self.fc1 = nn.Linear(self.input_dim, 400)
        self.fc21 = nn.Linear(400, self.latent_dim)
        self.fc22 = nn.Linear(400, self.latent_dim)
        self.fc3 = nn.Linear(self.latent_dim, 400)
        self.fc4 = nn.Linear(400, self.input_dim)

    def encode(self, x):
        """Encode a torch tensor (batch_size, input_size)"""
        x = x.view(-1, self.input_dim)
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def _reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        """Expand a latent memory, to input_size."""
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def sample(self, n, device=None):
        """Use noise to sample n images from latent space."""
        with torch.no_grad():
            x = torch.randn(n, self.latent_dim)
            x = x
            samples = self.decode(x)
            return samples

    def forward(self, x):
        """Get a reconstructed image"""
        mu, logvar = self.encode(x)
        z = self._reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


def get_inputs():
    return [torch.rand([4, 784])]


def get_init_inputs():
    return [[], {}]
