import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data


class Autoencoder(nn.Module):

    def __init__(self):
        super(Autoencoder, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, padding=2, kernel_size=5)
        self.maxpool1 = nn.MaxPool2d(4, stride=1, return_indices=True)
        self.conv2 = nn.Conv2d(6, 16, padding=2, kernel_size=5)
        self.maxpool2 = nn.MaxPool2d(2, stride=1, return_indices=True)
        self.unpool2 = nn.MaxUnpool2d(2, stride=1)
        self.unconv2 = nn.ConvTranspose2d(16, 6, padding=2, kernel_size=5)
        self.unpool1 = nn.MaxUnpool2d(4, stride=1)
        self.unconv1 = nn.ConvTranspose2d(6, 3, padding=2, kernel_size=5)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x, indices1 = self.maxpool1(x)
        x = self.conv2(x)
        x = F.relu(x)
        x, indices2 = self.maxpool2(x)
        x = self.unpool2(x, indices2)
        x = F.relu(x)
        x = self.unconv2(x)
        x = self.unpool1(x, indices1)
        x = F.relu(x)
        x = self.unconv1(x)
        return x

    def encoder(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x, indices1 = self.maxpool1(x)
        x = self.conv2(x)
        x = F.relu(x)
        x, indices2 = self.maxpool2(x)
        return x, indices1, indices2

    def decoder(self, x, indices1, indices2):
        x = self.conv1(x)
        x = F.relu(x)
        x, indices1 = self.maxpool1(x)
        x = self.conv2(x)
        x = F.relu(x)
        x, indices2 = self.maxpool2(x)
        return x, indices1, indices2


def get_inputs():
    return [torch.rand([4, 3, 64, 64])]


def get_init_inputs():
    return [[], {}]
