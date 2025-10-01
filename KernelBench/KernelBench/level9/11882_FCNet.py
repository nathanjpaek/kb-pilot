import torch
import torch.nn as nn
import torch.nn.functional as F


class FCNet(nn.Module):
    """ fully-connected neural network """

    def __init__(self):
        super(FCNet, self).__init__()
        self.fc1 = nn.Linear(784, 400)
        self.fc2 = nn.Linear(400, 200)
        self.fc3 = nn.Linear(200, 100)
        self.fc4 = nn.Linear(100, 10)

    def forward(self, x):
        x = x.view(-1, 784)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return F.log_softmax(x, dim=1)


def get_inputs():
    return [torch.rand([4, 784])]


def get_init_inputs():
    return [[], {}]
