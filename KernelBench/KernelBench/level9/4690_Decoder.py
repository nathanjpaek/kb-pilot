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


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 4, 'latent_size': 4}]
