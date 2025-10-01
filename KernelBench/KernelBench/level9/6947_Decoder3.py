import torch
import torch.nn as nn


class Decoder3(nn.Module):

    def __init__(self, M, H, D):
        super().__init__()
        self.D = D
        self.M = M
        self.H = H
        self.dec1 = nn.Linear(in_features=self.M, out_features=self.H)
        self.dec2 = nn.Linear(in_features=self.H, out_features=self.H * 2)
        self.dec3 = nn.Linear(in_features=self.H * 2, out_features=self.H * 3)
        self.dec4 = nn.Linear(in_features=self.H * 3, out_features=self.D)
        self.log_scale = nn.Parameter(torch.Tensor([0.0]))

    def forward(self, Z):
        Z = self.dec1(Z)
        Z = nn.functional.relu(Z)
        Z = self.dec2(Z)
        Z = nn.functional.relu(Z)
        Z = self.dec3(Z)
        Z = nn.functional.relu(Z)
        mu = self.dec4(Z)
        std = torch.exp(self.log_scale)
        return mu, std


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'M': 4, 'H': 4, 'D': 4}]
