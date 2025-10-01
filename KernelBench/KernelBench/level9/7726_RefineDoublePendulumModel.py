import torch
import numpy as np
import torch.nn as nn


class SirenLayer(nn.Module):

    def __init__(self, in_f, out_f, w0=30, is_first=False, is_last=False):
        super().__init__()
        self.in_f = in_f
        self.w0 = w0
        self.linear = nn.Linear(in_f, out_f)
        self.is_first = is_first
        self.is_last = is_last
        self.init_weights()

    def init_weights(self):
        b = 1 / self.in_f if self.is_first else np.sqrt(6 / self.in_f
            ) / self.w0
        with torch.no_grad():
            self.linear.weight.uniform_(-b, b)

    def forward(self, x):
        x = self.linear(x)
        return x if self.is_last else torch.sin(self.w0 * x)


class RefineDoublePendulumModel(torch.nn.Module):

    def __init__(self, in_channels):
        super(RefineDoublePendulumModel, self).__init__()
        self.layer1 = SirenLayer(in_channels, 128, is_first=True)
        self.layer2 = SirenLayer(128, 64)
        self.layer3 = SirenLayer(64, 32)
        self.layer4 = SirenLayer(32, 4)
        self.layer5 = SirenLayer(4, 32)
        self.layer6 = SirenLayer(32, 64)
        self.layer7 = SirenLayer(64, 128)
        self.layer8 = SirenLayer(128, in_channels, is_last=True)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        latent = self.layer4(x)
        x = self.layer5(latent)
        x = self.layer6(x)
        x = self.layer7(x)
        x = self.layer8(x)
        return x, latent


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 4}]
