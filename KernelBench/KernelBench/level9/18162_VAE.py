import torch
import torch.utils.data
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import *
from sklearn.metrics import *


class VAE(nn.Module):

    def __init__(self):
        super(VAE, self).__init__()
        self.encoder_fc = nn.Linear(784, 400)
        self.mean_fc = nn.Linear(400, 20)
        self.logvar_fc = nn.Linear(400, 20)
        self.prefinal_fc = nn.Linear(20, 400)
        self.final_fc = nn.Linear(400, 784)

    def encoder(self, x):
        encoded = torch.relu(self.encoder_fc(x))
        mu = self.mean_fc(encoded)
        log_var = self.logvar_fc(encoded)
        return mu, log_var

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def decoder(self, z):
        decoded = F.relu(self.prefinal_fc(z))
        return torch.sigmoid(self.final_fc(decoded))

    def forward(self, x):
        x = x.view(-1, 784)
        mu, log_var = self.encoder(x)
        z = self.reparameterize(mu, log_var)
        return self.decoder(z), mu, log_var


def get_inputs():
    return [torch.rand([4, 784])]


def get_init_inputs():
    return [[], {}]
