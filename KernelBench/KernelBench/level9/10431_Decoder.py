import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F


class Decoder(nn.Module):
    """ VAE decoder """

    def __init__(self, img_channels, latent_size):
        super(Decoder, self).__init__()
        self.latent_size = latent_size
        self.img_channels = img_channels
        self.fc1 = nn.Linear(latent_size, 6144)
        self.deconv1 = nn.ConvTranspose2d(128, 128, [1, 8], stride=[1, 2])
        self.deconv2 = nn.ConvTranspose2d(128, 64, [1, 16], stride=[1, 2])
        self.deconv3 = nn.ConvTranspose2d(64, 32, [1, 16], stride=[1, 2])
        self.deconv4 = nn.ConvTranspose2d(32, img_channels, [1, 22], stride
            =[1, 2])

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = x.view(x.size(0), -1, 16, 3)
        x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))
        x = F.relu(self.deconv3(x))
        reconstruction = torch.sigmoid(self.deconv4(x))
        return reconstruction


def get_inputs():
    return [torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'img_channels': 4, 'latent_size': 4}]
