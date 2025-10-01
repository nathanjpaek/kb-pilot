import torch
import torch.nn as nn
import torch.nn.functional as F


class ECToCA3(nn.Module):

    def __init__(self, D_in, D_out):
        super(ECToCA3, self).__init__()
        self.fc1 = nn.Linear(D_in, 800)
        self.fc2 = nn.Linear(800, D_out)

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x), 0.1618)
        x = torch.sigmoid(self.fc2(x))
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'D_in': 4, 'D_out': 4}]
