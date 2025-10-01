import torch
import torch.nn as nn


class AutoEncoder(nn.Module):

    def __init__(self, channels):
        """
        param:
            channels: a list containing all channels in the network.
        """
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential()
        for i in range(len(channels) - 1):
            self.encoder.add_module('fc%d' % (i + 1), nn.Linear(channels[i],
                channels[i + 1]))
            self.encoder.add_module('relu%d' % (i + 1), nn.ReLU(True))
        channels = list(reversed(channels))
        self.decoder = nn.Sequential()
        for i in range(len(channels) - 1):
            self.decoder.add_module('deconv%d' % (i + 1), nn.Linear(
                channels[i], channels[i + 1]))
            self.decoder.add_module('relu%d' % i, nn.ReLU(True))

    def forward(self, x):
        hidden = self.encoder(x)
        y = self.decoder(hidden)
        return y


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'channels': [4, 4]}]
