import torch
import torch.nn.functional as F
import torch.nn as nn


class IntrinsicsModel(nn.Module):

    def __init__(self, n, H, W):
        super(IntrinsicsModel, self).__init__()
        self.skew_scale = 0.001
        self.fc1 = nn.Linear(n, n)
        self.fc2 = nn.Linear(n, n)
        self.fc3 = nn.Linear(n, 5)
        self.H = H
        self.W = W

    def forward(self, x):
        x = F.elu(self.fc1(x))
        x = F.tanh(self.fc2(x))
        x = self.fc3(x)
        intrinsics = torch.cat((F.softplus(x[:, :1]) * self.W, F.softplus(x
            [:, 1:2]) * self.H, F.sigmoid(x[:, 2:3]) * self.W, F.sigmoid(x[
            :, 3:4]) * self.H, x[:, 4:] * self.skew_scale), dim=1)
        return intrinsics


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'n': 4, 'H': 4, 'W': 4}]
