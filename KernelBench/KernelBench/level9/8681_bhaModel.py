import torch
import torch.nn as nn
import torch.nn.functional as F


class bhaModel(nn.Module):

    def __init__(self, inShape, outShape):
        super().__init__()
        self.inShape = inShape
        self.outShape = outShape
        self.fc1 = nn.Linear(self.inShape, 32)
        self.fc2 = nn.Linear(32, 64)
        self.fc3 = nn.Linear(64, 128)
        self.fc4 = nn.Linear(128, 256)
        self.fc5 = nn.Linear(256, self.outShape)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'inShape': 4, 'outShape': 4}]
