import torch
import torch.nn as nn


class NNSmall(nn.Module):
    """
    Sammut et. al. benchmark
    """

    def __init__(self, state_dim, output_dim):
        super(NNSmall, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(128, 128)
        self.relu2 = nn.ReLU()
        self.fc21 = nn.Linear(128, 32)
        self.relu21 = nn.ReLU()
        self.fc22 = nn.Linear(32, 32)
        self.relu22 = nn.ReLU()
        self.fc23 = nn.Linear(32, 32)
        self.relu23 = nn.ReLU()
        self.fc3 = nn.Linear(32, output_dim)
        self.soft = nn.LogSoftmax(dim=0)

    def forward(self, x):
        """
        forward pass
        :param x: i_minus_j or vice versa
        :return:
        """
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc21(x)
        x = self.relu21(x)
        x = self.fc22(x)
        x = self.relu22(x)
        x = self.fc23(x)
        x = self.relu23(x)
        x = self.fc3(x)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'state_dim': 4, 'output_dim': 4}]
