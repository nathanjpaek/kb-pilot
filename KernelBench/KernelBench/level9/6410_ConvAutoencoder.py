import torch
import torch.nn.functional as F
from torch import nn
import torch.utils.data


class ConvAutoencoder(nn.Module):

    def __init__(self):
        super(ConvAutoencoder, self).__init__()
        self.conv1 = nn.Conv2d(12, 16, 3)
        self.conv2 = nn.Conv2d(16, 4, 3)
        self.t_conv1 = nn.ConvTranspose2d(4, 16, 3)
        self.t_conv2 = nn.ConvTranspose2d(16, 12, 3)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.t_conv1(x))
        x = F.sigmoid(self.t_conv2(x))
        return x


def get_inputs():
    return [torch.rand([4, 12, 64, 64])]


def get_init_inputs():
    return [[], {}]
