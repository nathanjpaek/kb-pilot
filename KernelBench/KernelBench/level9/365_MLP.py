import torch
from torch import nn
from torch.nn import functional as F


class MLP(nn.Module):
    """
    fc1, fc2
        two fully connected layers
    fc3
        output layer
    relu
        activation function for hidden layers
    sigmoid
        activation function for output layer
    """

    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(392, 160)
        self.fc2 = nn.Linear(160, 64)
        self.fc3 = nn.Linear(64, 2)

    def forward(self, x):
        x = x.view(-1, 392)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x


def get_inputs():
    return [torch.rand([4, 392])]


def get_init_inputs():
    return [[], {}]
