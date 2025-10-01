import torch
import torch.utils.data
from torch import nn


class CVAE(nn.Module):

    def __init__(self, conditional_size, hidden_size, latent_size):
        super(CVAE, self).__init__()
        self.fc1 = nn.Linear(28 * 28 + conditional_size, hidden_size)
        self.fc21 = nn.Linear(hidden_size, latent_size)
        self.fc22 = nn.Linear(hidden_size, latent_size)
        self.fc3 = nn.Linear(latent_size + conditional_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, 28 * 28)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def encode(self, x, c):
        inputs = torch.cat([x, c], 1)
        h1 = self.relu(self.fc1(inputs))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, z, c):
        inputs = torch.cat([z, c], 1)
        h3 = self.relu(self.fc3(inputs))
        return self.sigmoid(self.fc4(h3))

    def forward(self, x, c):
        mu, logvar = self.encode(x.view(-1, 28 * 28), c)
        z = self.reparameterize(mu, logvar)
        return self.decode(z, c), mu, logvar


def get_inputs():
    return [torch.rand([4, 784]), torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'conditional_size': 4, 'hidden_size': 4, 'latent_size': 4}]
