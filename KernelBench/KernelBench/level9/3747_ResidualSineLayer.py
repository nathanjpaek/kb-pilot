import torch
import numpy as np
import torch.nn as nn
import torch.utils.data


class ResidualSineLayer(nn.Module):
    """
    From Lu & Berger 2021, Compressive Neural Representations of Volumetric Scalar Fields
    https://github.com/matthewberger/neurcomp/blob/main/siren.py
    """

    def __init__(self, features: 'int', bias=True, ave_first=False,
        ave_second=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.features = features
        self.linear_1 = nn.Linear(features, features, bias=bias)
        self.linear_2 = nn.Linear(features, features, bias=bias)
        self.weight_1 = 0.5 if ave_first else 1
        self.weight_2 = 0.5 if ave_second else 1
        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            self.linear_1.weight.uniform_(-np.sqrt(6 / self.features) /
                self.omega_0, np.sqrt(6 / self.features) / self.omega_0)
            self.linear_2.weight.uniform_(-np.sqrt(6 / self.features) /
                self.omega_0, np.sqrt(6 / self.features) / self.omega_0)

    def forward(self, input):
        sine_1 = torch.sin(self.omega_0 * self.linear_1(self.weight_1 * input))
        sine_2 = torch.sin(self.omega_0 * self.linear_2(sine_1))
        return self.weight_2 * (input + sine_2)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'features': 4}]
