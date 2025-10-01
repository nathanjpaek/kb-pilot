import torch
from torch import nn
from torch.nn import functional as F


class SiameseMLP(nn.Module):
    """
    basic structure similar to the MLP
    input is splited into two 1*14*14 images for separating training, share the same parameters
    """

    def __init__(self):
        super(SiameseMLP, self).__init__()
        self.fc1 = nn.Linear(196, 160)
        self.fc2 = nn.Linear(160, 10)
        self.fc3 = nn.Linear(20, 2)

    def forward(self, x1, x2):
        x1 = x1.view(-1, 196)
        x1 = F.relu(self.fc1(x1))
        x1 = self.fc2(x1)
        x1 = F.relu(x1)
        x2 = x2.view(-1, 196)
        x2 = F.relu(self.fc1(x2))
        x2 = self.fc2(x2)
        x2 = F.relu(x2)
        x = torch.cat([x1, x2], dim=1)
        x = torch.sigmoid(self.fc3(x))
        return x


def get_inputs():
    return [torch.rand([4, 196]), torch.rand([4, 196])]


def get_init_inputs():
    return [[], {}]
