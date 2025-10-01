import torch
from torch import nn
from torch.nn import functional as F
import torch.utils.data


class Decoder(nn.Module):
    """ VAE decoder """

    def __init__(self, in_channels, latent_size):
        super(Decoder, self).__init__()
        self.latent_size = latent_size
        self.in_channels = in_channels
        self.fc_dec1 = nn.Linear(latent_size, 200)
        self.fc_dec2 = nn.Linear(200, 200)
        self.fc_dec3 = nn.Linear(200, self.in_channels)

    def forward(self, x):
        x = F.relu(self.fc_dec1(x))
        x = F.relu(self.fc_dec2(x))
        x = F.relu(self.fc_dec3(x))
        reconstruction = F.sigmoid(x)
        return reconstruction


class Encoder(nn.Module):
    """ VAE encoder """

    def __init__(self, in_channels, latent_size):
        super(Encoder, self).__init__()
        self.latent_size = latent_size
        self.in_channels = in_channels
        self.fc_enc1 = nn.Linear(self.in_channels, 200)
        self.fc_enc2 = nn.Linear(200, 200)
        self.fc_mu = nn.Linear(200, latent_size)
        self.fc_logsigma = nn.Linear(200, latent_size)

    def forward(self, x):
        x = F.relu(self.fc_enc1(x))
        x = F.relu(self.fc_enc2(x))
        mu = self.fc_mu(x)
        logsigma = self.fc_logsigma(x)
        return mu, logsigma


class VAE(nn.Module):
    """ Variational Autoencoder """

    def __init__(self, in_channels, latent_size):
        super(VAE, self).__init__()
        self.encoder = Encoder(in_channels, latent_size)
        self.decoder = Decoder(in_channels, latent_size)

    def forward(self, x):
        mu, logsigma = self.encoder(x)
        sigma = logsigma.exp()
        eps = torch.randn_like(sigma)
        z = eps.mul(sigma).add_(mu)
        recon_x = self.decoder(z)
        return recon_x, mu, logsigma


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 4, 'latent_size': 4}]
