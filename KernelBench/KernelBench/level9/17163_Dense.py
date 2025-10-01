import torch
import torch.nn as nn
import torch.nn.functional as functions


class Dense(nn.Module):

    def __init__(self):
        super(Dense, self).__init__()
        self.fc1 = nn.Linear(6 * 7, 32)
        self.fc2 = nn.Linear(32, 16)
        self.probhead = nn.Linear(16, 7)
        self.valuehead = nn.Linear(16, 1)
        self.soft = nn.Softmax(dim=1)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = x.view(-1, 6 * 7)
        x = functions.relu(self.fc1(x))
        x = functions.relu(self.fc2(x))
        P = self.soft(self.probhead(x))
        v = self.tanh(self.valuehead(x))
        return P, v


def get_inputs():
    return [torch.rand([4, 42])]


def get_init_inputs():
    return [[], {}]
