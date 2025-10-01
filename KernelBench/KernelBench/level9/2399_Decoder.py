import torch
import torch.nn as nn


class Decoder(nn.Module):

    def __init__(self, sampling_rate=16000.0):
        super(Decoder, self).__init__()
        self.sampling_rate = sampling_rate
        self.upsa1 = torch.nn.Upsample(int(sampling_rate / 2))
        self.conv3 = torch.nn.Conv1d(128, 64, 3, padding=1)
        self.upsa2 = torch.nn.Upsample(int(sampling_rate))
        self.conv4 = torch.nn.Conv1d(64, 1, 3, padding=1)

    def forward(self, x):
        x = self.upsa1(x)
        x = self.conv3(x)
        x = torch.relu(x)
        x = self.upsa2(x)
        x = self.conv4(x)
        x = torch.tanh(x)
        return x


def get_inputs():
    return [torch.rand([4, 128, 4])]


def get_init_inputs():
    return [[], {}]
