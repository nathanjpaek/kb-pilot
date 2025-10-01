import torch
from torch import nn


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


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'latent_channel_dim': 4}]
