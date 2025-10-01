import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleNN(nn.Module):

    def __init__(self, in_values, out_values):
        super().__init__()
        self.dense1 = nn.Linear(in_values, 12673)
        self.drop1 = nn.Dropout()
        self.dense2 = nn.Linear(12673, 4000)
        self.drop2 = nn.Dropout()
        self.dense3 = nn.Linear(4000, 500)
        self.drop3 = nn.Dropout()
        self.last_dense = nn.Linear(500, out_values)

    def forward(self, x):
        x = F.relu(self.dense1(x))
        x = self.drop1(x)
        x = F.relu(self.dense2(x))
        x = self.drop2(x)
        x = F.relu(self.dense3(x))
        x = self.drop3(x)
        x = self.last_dense(x)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_values': 4, 'out_values': 4}]
