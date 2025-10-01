import torch
from torch.nn import functional as F
from torch import nn
import torch.nn


class VAE(nn.Module):

    def __init__(self, in_ch, out_ch, hidden_ch=128):
        super(VAE, self).__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.fc1 = nn.Linear(in_ch, hidden_ch)
        self.fc21 = nn.Linear(hidden_ch, 20)
        self.fc22 = nn.Linear(hidden_ch, 20)
        self.fc3 = nn.Linear(20, hidden_ch)
        self.fc4 = nn.Linear(hidden_ch, out_ch)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, self.in_ch))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_ch': 4, 'out_ch': 4}]
