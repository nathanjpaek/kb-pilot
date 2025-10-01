import torch
import torch.nn as nn
import torch.nn.functional as F


class spectral_model(nn.Module):

    def __init__(self, num_classes):
        super(spectral_model, self).__init__()
        self.mlp1 = nn.Conv1d(6, 64, 1)
        self.mlp2 = nn.Conv1d(64, 128, 1)
        self.mlp3 = nn.Conv1d(128, 256, 1)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = F.relu(self.mlp1(x))
        x = F.relu(self.mlp2(x))
        x = F.relu(self.mlp3(x))
        x = torch.max(x, 2, keepdim=True)[0]
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def get_inputs():
    return [torch.rand([4, 6, 6])]


def get_init_inputs():
    return [[], {'num_classes': 4}]
