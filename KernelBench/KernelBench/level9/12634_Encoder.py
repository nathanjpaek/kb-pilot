import torch
import torch.nn.functional as F
from torch import nn


class Encoder(torch.nn.Module):

    def __init__(self, X_dim, h_dim, Z_dim):
        super(Encoder, self).__init__()
        self.hidden1 = torch.nn.Linear(X_dim, X_dim)
        self.hidden2 = torch.nn.Linear(X_dim, h_dim)
        self.hidden3 = torch.nn.Linear(h_dim, int(h_dim / 2))
        self.hidden4 = torch.nn.Linear(int(h_dim / 2), int(h_dim / 4))
        self.out1 = torch.nn.Linear(int(h_dim / 4), Z_dim)
        self.out2 = torch.nn.Linear(int(h_dim / 4), Z_dim)

    def forward(self, X):
        h = nn.Dropout(p=0.0)(F.selu(self.hidden1(X)))
        h = nn.Dropout(p=0.0)(F.selu(self.hidden2(h)))
        h = nn.Dropout(p=0.0)(F.selu(self.hidden3(h)))
        h = nn.Dropout(p=0.0)(F.selu(self.hidden4(h)))
        mu = self.out1(h)
        log_var = self.out2(h)
        return mu, log_var


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'X_dim': 4, 'h_dim': 4, 'Z_dim': 4}]
