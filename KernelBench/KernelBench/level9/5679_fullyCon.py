import torch
import torch.nn as nn
import torch.nn.functional as F


class fullyCon(nn.Module):

    def __init__(self):
        super(fullyCon, self).__init__()
        self.fc1 = nn.Linear(448 * 3 * 448, 500)
        self.fc2 = nn.Linear(500, 100)
        self.fc3 = nn.Linear(100, 5)

    def forward(self, x):
        x = x.view(-1, 448 * 3 * 448)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def get_inputs():
    return [torch.rand([4, 602112])]


def get_init_inputs():
    return [[], {}]
