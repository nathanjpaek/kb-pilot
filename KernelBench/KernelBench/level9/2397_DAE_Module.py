import torch
import torch.nn as nn


class Encoder(nn.Module):

    def __init__(self):
        super(Encoder, self).__init__()
        self.conv1 = torch.nn.Conv1d(1, 64, 3, padding=1)
        self.maxp1 = torch.nn.MaxPool1d(2, padding=0)
        self.conv2 = torch.nn.Conv1d(64, 128, 3, padding=1)
        self.maxp2 = torch.nn.MaxPool1d(2, padding=0)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.relu(x)
        x = self.maxp1(x)
        x = self.conv2(x)
        x = torch.relu(x)
        x = self.maxp2(x)
        return x


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


class DAE_Module(nn.Module):

    def __init__(self, sampling_rate=16000.0):
        super(DAE_Module, self).__init__()
        self.sampling_rate = int(sampling_rate)
        self.encoder = Encoder()
        self.decoder = Decoder(sampling_rate=self.sampling_rate)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


def get_inputs():
    return [torch.rand([4, 1, 64])]


def get_init_inputs():
    return [[], {}]
