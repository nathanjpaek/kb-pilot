import torch
import torch.utils.data
from torch import nn
from torch.nn import functional as F


class VAE_genes(nn.Module):

    def __init__(self):
        super(VAE_genes, self).__init__()
        self.input_linear = nn.Linear(907, 500)
        self.enc_middle = nn.Linear(500, 100)
        self.enc_1 = nn.Linear(100, 5)
        self.enc_2 = nn.Linear(100, 5)
        self.dec_0 = nn.Linear(5, 100)
        self.dec_middle = nn.Linear(100, 500)
        self.output_linear = nn.Linear(500, 907)

    def encode(self, x):
        h1 = F.relu(self.input_linear(x))
        h2 = F.relu(self.enc_middle(h1))
        return self.enc_1(h2), self.enc_2(h2)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def decode(self, z):
        h3 = F.relu(self.dec_0(z))
        h4 = F.relu(self.dec_middle(h3))
        return torch.sigmoid(self.output_linear(h4))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 907))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


def get_inputs():
    return [torch.rand([4, 907])]


def get_init_inputs():
    return [[], {}]
