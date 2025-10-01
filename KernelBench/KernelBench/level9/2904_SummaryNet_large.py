import torch
import torch.nn as nn
import torch.nn.functional as F


class SummaryNet_large(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=2, out_channels=20, kernel_size=
            3, padding=2)
        self.pool = nn.MaxPool1d(kernel_size=5, stride=5)
        self.fc = nn.Linear(in_features=20 * 5 * 2, out_features=8)

    def forward(self, x):
        x = x.view(-1, 2, 51)
        x = self.pool(F.relu(self.conv1(x)))
        x = x.view(-1, 20 * 5 * 2)
        x = F.relu(self.fc(x))
        return x


def get_inputs():
    return [torch.rand([4, 2, 51])]


def get_init_inputs():
    return [[], {}]
