import torch
from torch import nn


class MLP_VAE(nn.Module):

    def __init__(self, ZDIMS):
        super().__init__()
        self.z_dims = ZDIMS
        self.fc1 = nn.Linear(1024, 400)
        self.relu = nn.ReLU()
        self.fc21 = nn.Linear(400, ZDIMS)
        self.fc22 = nn.Linear(400, ZDIMS)
        self.fc3 = nn.Linear(ZDIMS, 400)
        self.fc4 = nn.Linear(400, 1024)

    def encoder(self, x):
        """
        Input vector x --> fully connected 1 --> RELU --> fully connected 21, fully connected 22
        
        Parameters
        ----------
        x: [batch size, 784], batch size number of digits of 28x28 pixels each
        
        Returns 
        -------
        (mu, logvar): ZDIMS mean units one for each latent dimension, ZDIMS variance units one for each 
        latent dimension
        """
        batch_size = x.shape[0]
        x = x.view(batch_size, -1)
        h1 = self.relu(self.fc1(x))
        mu = self.fc21(h1)
        logvar = self.fc22(h1)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decoder(self, z):
        h3 = self.relu(self.fc3(z))
        return self.fc4(h3)

    def forward(self, x):
        mu, logvar = self.encoder(x.view(-1, 1024))
        z = self.reparameterize(mu, logvar)
        reconstruction_x = self.decoder(z)
        return reconstruction_x, mu, logvar

    def sample(self, n):
        with torch.no_grad():
            z = torch.randn(n, self.z_dims)
            z = z
            samples = self.decoder(z)
            samples = torch.clamp(samples, 0, 1)
        return samples.cpu().numpy()


def get_inputs():
    return [torch.rand([4, 1024])]


def get_init_inputs():
    return [[], {'ZDIMS': 4}]
