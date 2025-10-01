import torch
import torch.nn as nn
import torch.nn.functional as F


class VAEEncoder(nn.Module):

    def __init__(self, z_size):
        super(VAEEncoder, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 4, stride=2)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 128, 4, stride=2)
        self.conv4 = nn.Conv2d(128, 256, 4, stride=2)
        self.fc_mu = nn.Linear(2 * 2 * 256, z_size)
        self.fc_log_var = nn.Linear(2 * 2 * 256, z_size)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = x.view(-1, 2 * 2 * 256)
        mu = self.fc_mu(x)
        log_var = self.fc_log_var(x)
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z, mu, log_var


def get_inputs():
    return [torch.rand([4, 3, 64, 64])]


def get_init_inputs():
    return [[], {'z_size': 4}]
