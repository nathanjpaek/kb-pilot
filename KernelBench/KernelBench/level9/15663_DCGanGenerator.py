import torch
import torch.nn as nn
from torch.nn import functional as F


class DCGanGenerator(nn.Module):

    def __init__(self, latent_dim):
        super().__init__()
        self.fc1 = nn.Linear(latent_dim, 2 * 2 * 512)
        self.conv1 = nn.ConvTranspose2d(512, 256, kernel_size=5, stride=1,
            padding=1, output_padding=0)
        self.conv2 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2,
            padding=1)
        self.conv3 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2,
            padding=1, output_padding=0)
        self.conv4 = nn.ConvTranspose2d(64, 3, kernel_size=5, stride=2,
            padding=3)

    def forward(self, input):
        x = self.fc1(input)
        x = x.view(x.size(0), 512, 2, 2)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        return torch.tanh(x)


def get_inputs():
    return [torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'latent_dim': 4}]
