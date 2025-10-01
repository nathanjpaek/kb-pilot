import torch
from torch import nn as nn
from torch import optim as optim
from torchvision import transforms as transforms


class MNISTGenerator(nn.Module):

    def __init__(self, latent_dim):
        super(MNISTGenerator, self).__init__()
        self.image_shape = 1, 28, 28
        self.latent_dim = latent_dim
        self.dense1 = nn.Linear(self.latent_dim, 128, True)
        self.dense2 = nn.Linear(128, 784, True)

    def forward(self, x):
        x = nn.functional.relu(self.dense1(x))
        x = nn.functional.sigmoid(self.dense2(x))
        return x.view(x.shape[0], *self.image_shape)


def get_inputs():
    return [torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'latent_dim': 4}]
