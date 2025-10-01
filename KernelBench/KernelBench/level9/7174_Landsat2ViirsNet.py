import torch
from torch import nn
from torch.nn import functional as F


class Landsat2ViirsNet(nn.Module):

    def __init__(self, latent_dim=64, init_channels=8, kernel_size=4,
        image_in_channels=3, image_out_channels=1):
        super(Landsat2ViirsNet, self).__init__()
        self.enc1 = nn.Conv2d(in_channels=image_in_channels, out_channels=
            init_channels, kernel_size=kernel_size, stride=4, padding=1,
            dilation=2)
        self.enc2 = nn.Conv2d(in_channels=init_channels, out_channels=
            init_channels * 2, kernel_size=kernel_size, stride=3, padding=1)
        self.enc3 = nn.Conv2d(in_channels=init_channels * 2, out_channels=
            init_channels * 4, kernel_size=kernel_size, stride=3, padding=1)
        self.enc4 = nn.Conv2d(in_channels=init_channels * 4, out_channels=
            64, kernel_size=7, stride=2, padding=0)
        self.fc1 = nn.Linear(64, 128)
        self.fc_mu = nn.Linear(128, latent_dim)
        self.fc_log_var = nn.Linear(128, latent_dim)
        self.fc2 = nn.Linear(latent_dim, 64)
        self.dec1 = nn.ConvTranspose2d(in_channels=64, out_channels=
            init_channels * 4, kernel_size=kernel_size, stride=1, padding=0)
        self.dec2 = nn.ConvTranspose2d(in_channels=init_channels * 4,
            out_channels=init_channels * 2, kernel_size=kernel_size, stride
            =2, padding=0)
        self.dec3 = nn.ConvTranspose2d(in_channels=init_channels * 2,
            out_channels=image_out_channels, kernel_size=kernel_size + 1,
            stride=2, padding=1)

    def reparameterize(self, mu, log_var):
        """
        :param mu: mean from the encoder's latent space
        :param log_var: log variance from the encoder's latent space
        """
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        sample = mu + eps * std
        return sample

    def forward(self, x):
        x = F.leaky_relu(self.enc1(x))
        x = F.leaky_relu(self.enc2(x))
        x = F.leaky_relu(self.enc3(x))
        x = F.leaky_relu(self.enc4(x))
        batch, _, _, _ = x.shape
        x = F.adaptive_avg_pool2d(x, 1).reshape(batch, -1)
        hidden = self.fc1(x)
        mu = self.fc_mu(hidden)
        log_var = self.fc_log_var(hidden)
        z = self.reparameterize(mu, log_var)
        z = self.fc2(z)
        z = z.view(-1, 64, 1, 1)
        x = F.leaky_relu(self.dec1(z))
        x = F.leaky_relu(self.dec2(x))
        reconstruction = torch.sigmoid(self.dec3(x))
        return reconstruction, mu, log_var


def get_inputs():
    return [torch.rand([4, 3, 256, 256])]


def get_init_inputs():
    return [[], {}]
