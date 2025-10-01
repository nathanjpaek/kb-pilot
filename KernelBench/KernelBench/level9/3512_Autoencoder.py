import torch
import torch.nn as nn


class Autoencoder(nn.Module):

    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Conv2d(1024, 128, kernel_size=1)
        self.decoder = nn.Conv2d(128, 1024, kernel_size=1)
        self.relu = nn.ReLU()

    def forward(self, local_f):
        encoded_f = self.encoder(local_f)
        decoded_f = self.decoder(encoded_f)
        decoded_f = self.relu(decoded_f)
        return encoded_f, decoded_f


def get_inputs():
    return [torch.rand([4, 1024, 64, 64])]


def get_init_inputs():
    return [[], {}]
