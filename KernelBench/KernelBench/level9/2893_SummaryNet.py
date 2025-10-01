import torch
import torch.nn as nn
import torch.nn.functional as F


class SummaryNet(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=6, kernel_size=5,
            padding=2)
        self.pool = nn.MaxPool1d(kernel_size=10, stride=10)
        self.fc = nn.Linear(in_features=6 * 10, out_features=8)

    def forward(self, x):
        x = x.view(-1, 1, 100)
        x = self.pool(F.relu(self.conv1(x)))
        x = x.view(-1, 6 * 10)
        x = F.relu(self.fc(x))
        return x


def get_inputs():
    return [torch.rand([4, 1, 100])]


def get_init_inputs():
    return [[], {}]
