import torch
import torch.nn as nn


class Encoder(nn.Module):

    def __init__(self, D, H, M):
        super().__init__()
        self.D = D
        self.M = M
        self.H = H
        self.enc1 = nn.Linear(in_features=self.D, out_features=self.H)
        self.enc2 = nn.Linear(in_features=self.H, out_features=self.H)
        self.enc3 = nn.Linear(in_features=self.H, out_features=self.M * 2)

    def forward(self, x):
        x = self.enc1(x)
        x = nn.functional.relu(x)
        x = self.enc2(x)
        x = nn.functional.relu(x)
        x = self.enc3(x)
        x = x.view(-1, 2, self.M)
        mu = x[:, 0, :]
        log_var = x[:, 1, :]
        std = torch.exp(log_var / 2)
        return mu, std


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'D': 4, 'H': 4, 'M': 4}]
