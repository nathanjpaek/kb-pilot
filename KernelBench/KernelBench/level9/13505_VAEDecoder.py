import torch
import torch.nn as nn
import torch.nn.functional as F


class VAEDecoder(nn.Module):

    def __init__(self, z_size):
        super(VAEDecoder, self).__init__()
        self.fc = nn.Linear(z_size, 4 * 256)
        self.deconv1 = nn.ConvTranspose2d(1024, 128, 5, stride=2)
        self.deconv2 = nn.ConvTranspose2d(128, 64, 5, stride=2)
        self.deconv3 = nn.ConvTranspose2d(64, 32, 6, stride=2)
        self.deconv4 = nn.ConvTranspose2d(32, 3, 6, stride=2)

    def forward(self, x):
        x = self.fc(x)
        x = x.view(-1, 4 * 256, 1, 1)
        x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))
        x = F.relu(self.deconv3(x))
        x = torch.sigmoid(self.deconv4(x))
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'z_size': 4}]
