import torch
import torch.nn as nn
import torch.nn.functional as F


class FC(nn.Module):
    """FC baseline implementation"""

    def __init__(self):
        super(FC, self).__init__()
        self.fc1 = nn.Linear(45 * 45, 1024)
        self.fc2 = nn.Linear(1024, 256)
        self.fc3 = nn.Linear(256, 64)
        self.fc4 = nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(-1, 45 * 45)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.fc2(x))
        x = F.dropout(x, training=self.training)
        x = self.fc3(x)
        x = F.dropout(x, training=self.training)
        x = self.fc4(x)
        return F.log_softmax(x, dim=1)


def get_inputs():
    return [torch.rand([4, 2025])]


def get_init_inputs():
    return [[], {}]
