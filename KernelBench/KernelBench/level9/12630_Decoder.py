import torch
import torch.nn.functional as F
from torch import nn


class Decoder(torch.nn.Module):

    def __init__(self, Z_dim, h_dim, X_dim):
        super(Decoder, self).__init__()
        self.hidden1 = torch.nn.Linear(Z_dim, int(h_dim / 4))
        self.hidden2 = torch.nn.Linear(int(h_dim / 4), int(h_dim / 2))
        self.hidden3 = torch.nn.Linear(int(h_dim / 2), h_dim)
        self.hidden4 = torch.nn.Linear(h_dim, X_dim)
        self.out = torch.nn.Linear(X_dim, X_dim)

    def forward(self, z):
        h = nn.Dropout(p=0.0)(F.selu(self.hidden1(z)))
        h = nn.Dropout(p=0.0)(F.selu(self.hidden2(h)))
        h = nn.Dropout(p=0.0)(F.selu(self.hidden3(h)))
        h = nn.Dropout(p=0.0)(F.selu(self.hidden4(h)))
        out = torch.sigmoid(self.out(h))
        return out


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'Z_dim': 4, 'h_dim': 4, 'X_dim': 4}]
