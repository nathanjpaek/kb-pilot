import torch
import torch.nn as nn
import torch.nn.functional as F


class _CAEAD(nn.Module):

    def __init__(self, input_size):
        super(_CAEAD, self).__init__()
        self.en_1 = nn.Conv1d(1, 64, 3, padding=1)
        self.pool1 = nn.MaxPool1d(2, 2)
        self.en_2 = nn.Conv1d(64, 32, 3, padding=1)
        self.pool2 = nn.MaxPool1d(2, 2)
        self.en_3 = nn.Conv1d(32, 8, 3, padding=1)
        self.pool3 = nn.MaxPool1d(2, 2)
        self.en_4 = nn.Conv1d(8, 4, 3, padding=1)
        self.pool4 = nn.MaxPool1d(2, 2)
        self.de_1 = nn.Conv1d(4, 8, 3, padding=1)
        self.de_2 = nn.Conv1d(8, 32, 3, padding=1)
        self.de_3 = nn.Conv1d(32, 64, 3, padding=1)
        self.de_4 = nn.Conv1d(64, 1, 3, padding=1)

    def forward(self, X):
        encoder = F.relu(self.en_1(X))
        encoder = self.pool1(encoder)
        encoder = F.relu(self.en_2(encoder))
        encoder = self.pool2(encoder)
        encoder = F.relu(self.en_3(encoder))
        encoder = self.pool3(encoder)
        encoder = F.relu(self.en_4(encoder))
        encoder = self.pool4(encoder)
        decoder = F.interpolate(encoder, scale_factor=2)
        decoder = F.relu(self.de_1(decoder))
        decoder = F.interpolate(decoder, scale_factor=2)
        decoder = F.relu(self.de_2(decoder))
        decoder = F.interpolate(decoder, scale_factor=2)
        decoder = F.relu(self.de_3(decoder))
        decoder = F.interpolate(decoder, scale_factor=2)
        decoder = self.de_4(decoder)
        return decoder


def get_inputs():
    return [torch.rand([4, 1, 64])]


def get_init_inputs():
    return [[], {'input_size': 4}]
