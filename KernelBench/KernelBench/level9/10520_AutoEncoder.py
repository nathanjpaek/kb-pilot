import torch
from torch import nn


class Encoder(nn.Module):

    def __init__(self, latent_channel_dim):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=
            (3, 3), stride=(1, 1), padding=(1, 1))
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=
            latent_channel_dim, kernel_size=(3, 3), stride=(1, 1), padding=
            (1, 1))
        self.pool1 = nn.MaxPool2d(2, stride=2)
        self.act1 = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.act1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.act1(x)
        x = self.pool1(x)
        return x


class Decoder(nn.Module):

    def __init__(self, latent_channel_dim):
        super(Decoder, self).__init__()
        self.t_conv1 = nn.ConvTranspose2d(in_channels=latent_channel_dim,
            out_channels=16, kernel_size=(2, 2), stride=(2, 2))
        self.t_conv2 = nn.ConvTranspose2d(in_channels=16, out_channels=3,
            kernel_size=(2, 2), stride=(2, 2))
        self.act1 = nn.ReLU()
        self.act2 = nn.Sigmoid()

    def forward(self, x):
        x = self.t_conv1(x)
        x = self.act1(x)
        x = self.t_conv2(x)
        x = self.act2(x)
        return x


class AutoEncoder(nn.Module):

    def __init__(self, latent_channel_dim):
        super(AutoEncoder, self).__init__()
        self.encoder = Encoder(latent_channel_dim=latent_channel_dim)
        self.decoder = Decoder(latent_channel_dim=latent_channel_dim)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


def get_inputs():
    return [torch.rand([4, 3, 64, 64])]


def get_init_inputs():
    return [[], {'latent_channel_dim': 4}]
