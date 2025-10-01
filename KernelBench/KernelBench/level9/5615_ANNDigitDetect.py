import torch
import torch.nn as nn
import torch.nn.functional as F


class ANNDigitDetect(nn.Module):

    def __init__(self):
        super(ANNDigitDetect, self).__init__()
        self.fc1 = nn.Linear(32 * 32, 120)
        self.fc2 = nn.Linear(120, 32)
        self.fc3 = nn.Linear(32, 10)

    def forward(self, x):
        x = x.view(-1, 32 * 32)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def get_inputs():
    return [torch.rand([4, 1024])]


def get_init_inputs():
    return [[], {}]
