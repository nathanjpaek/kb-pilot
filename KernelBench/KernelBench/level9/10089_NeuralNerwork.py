import torch
import torch.nn as nn
import torch.nn.functional as F


class NeuralNerwork(nn.Module):
    """ Construct a ReLU-activated NN, set Bias to False
        Four hidden layers with sizes [1000, 1000, 500, 200]
        Features = 784, Targets = 10 classes
    """

    def __init__(self, features, targets):
        super(NeuralNerwork, self).__init__()
        self.fc1 = nn.Linear(features, 1000, bias=False)
        self.fc2 = nn.Linear(1000, 1000, bias=False)
        self.fc3 = nn.Linear(1000, 500, bias=False)
        self.fc4 = nn.Linear(500, 200, bias=False)
        self.fc5 = nn.Linear(200, targets, bias=False)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        return F.relu(self.fc5(x))


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'features': 4, 'targets': 4}]
