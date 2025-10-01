import torch
import torch.nn as nn
import torch.nn.functional as F


class ContextgenCNN(nn.Module):

    def __init__(self, latent_dim=1024):
        super(ContextgenCNN, self).__init__()
        self.latent_dim = latent_dim
        self.conv1 = nn.Conv2d(self.latent_dim, self.latent_dim // 4, 1,
            stride=1, padding=0, bias=False)
        self.conv2 = nn.Conv2d(self.latent_dim // 4, self.latent_dim // 4, 
            2, stride=1, padding=1, bias=False)
        self.conv3 = nn.Conv2d(self.latent_dim // 4, self.latent_dim // 2, 
            2, stride=1, padding=0, bias=False)
        self.conv4 = nn.Conv2d(self.latent_dim // 2, self.latent_dim, 1,
            stride=1, padding=0, bias=False)

    def forward(self, x):
        x = F.elu(self.conv1(x))
        x = F.elu(self.conv2(x))
        x = F.elu(self.conv3(x))
        x = F.elu(self.conv4(x))
        return x

    def train_order_block_ids(self):
        return [[0, 3]]


def get_inputs():
    return [torch.rand([4, 1024, 64, 64])]


def get_init_inputs():
    return [[], {}]
