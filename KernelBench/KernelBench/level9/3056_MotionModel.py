import torch
import torch.nn.functional as F
import torch.nn as nn


class MotionModel(nn.Module):

    def __init__(self, n):
        super(MotionModel, self).__init__()
        self.rotation_scale = 0.01
        self.fc1 = nn.Linear(n, n)
        self.fc2 = nn.Linear(n, n)
        self.fc3 = nn.Linear(n, n)
        self.rotation = nn.Linear(n, 3)
        self.translation = nn.Linear(n, 3)

    def forward(self, x):
        x = F.elu(self.fc1(x))
        translation = F.tanh(self.translation(x))
        x = F.elu(self.fc2(x))
        x = F.tanh(self.fc3(x))
        rotation = self.rotation(x) * self.rotation_scale
        return rotation, translation


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'n': 4}]
