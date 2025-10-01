import torch
from torch import nn
import torch.nn.functional as F


class FFN(nn.Module):

    def __init__(self, d):
        super().__init__()
        self.fc_1 = nn.Linear(2 * d, 4 * d)
        self.drop = nn.Dropout(0.1)
        self.fc_2 = nn.Linear(4 * d, d)

    def forward(self, x_1, x_2):
        x = self.fc_1(torch.cat((x_1, x_2), 1))
        x = F.relu(x)
        x = self.drop(x)
        x = self.fc_2(x)
        return x


def get_inputs():
    return [torch.rand([4, 4]), torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'d': 4}]
