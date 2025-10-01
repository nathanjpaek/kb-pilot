import torch
import torch.nn as nn


class Decoder(nn.Module):

    def __init__(self, latent_size, m):
        super(Decoder, self).__init__()
        self.latent_size = latent_size
        self.fc = nn.Linear(latent_size, m)
        self.deconv1 = nn.ConvTranspose2d(m, 128, 5, stride=2)
        self.deconv2 = nn.ConvTranspose2d(128, 64, 5, stride=2)
        self.deconv3 = nn.ConvTranspose2d(64, 32, 6, stride=2)
        self.deconv4 = nn.ConvTranspose2d(32, 1, 6, stride=2)

    def forward(self, x):
        x = torch.relu(self.fc(x))
        x = x.unsqueeze(-1).unsqueeze(-1)
        x = torch.relu(self.deconv1(x))
        x = torch.relu(self.deconv2(x))
        x = torch.relu(self.deconv3(x))
        reconstr = torch.sigmoid(self.deconv4(x))
        return reconstr


def get_inputs():
    return [torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'latent_size': 4, 'm': 4}]
