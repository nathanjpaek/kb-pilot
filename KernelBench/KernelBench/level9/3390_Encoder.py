import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):

    def __init__(self, sample_size, condition_size, hidden_size):
        super().__init__()
        self.fc1 = nn.Linear(sample_size + condition_size, hidden_size)
        self.fc2 = nn.Dropout(p=0.5)
        self.fc3 = nn.Linear(hidden_size, hidden_size)

    def forward(self, x, c):
        x = torch.cat([x, c], 1)
        p_x = F.relu(self.fc1(x))
        p_x = self.fc2(p_x)
        p_x = F.relu(self.fc3(p_x))
        return p_x


def get_inputs():
    return [torch.rand([4, 4]), torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'sample_size': 4, 'condition_size': 4, 'hidden_size': 4}]
