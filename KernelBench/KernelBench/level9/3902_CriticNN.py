import torch
import torch.optim as optim
from torch import nn
from torch.nn import functional as F


class CriticNN(nn.Module):

    def __init__(self, in_channels=3):
        super(CriticNN, self).__init__()
        self.fc1 = nn.Linear(4, 64)
        self.fc2 = nn.Linear(64, 1)
        self.optimizer = optim.Adam(self.parameters(), lr=0.0001)
        None

    def forward(self, x):
        x = F.layer_norm(x, x.size())
        x = F.leaky_relu(self.fc1(x))
        x = F.layer_norm(x, x.size())
        x = self.fc2(x)
        return x

    def init_weights(self, m):
        if type(m) == nn.Linear:
            None
            m.weight.data.fill_(0)
            m.bias.data.fill_(0)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
