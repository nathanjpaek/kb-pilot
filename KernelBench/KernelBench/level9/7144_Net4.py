import torch
from torch import nn
from torch.nn.init import kaiming_normal
from torch.nn.init import normal


def weights_init(m):
    if isinstance(m, (nn.Conv1d, nn.Linear)):
        kaiming_normal(m.weight.data)
        try:
            kaiming_normal(m.bias.data)
        except ValueError:
            normal(m.bias.data)


class Net4(nn.Module):
    """
    Net4 is a neural network consisting of five hidden layers with sizes 400,
    300, 200, 100 and 60
    """
    hidden1 = 400
    hidden2 = 300
    hidden3 = 200
    hidden4 = 100
    hidden5 = 60

    def __init__(self, input_size):
        super(Net4, self).__init__()
        self.fc1 = nn.Linear(input_size, self.hidden1)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(self.hidden1, self.hidden2)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(self.hidden2, self.hidden3)
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(self.hidden3, self.hidden4)
        self.relu4 = nn.ReLU()
        self.fc5 = nn.Linear(self.hidden4, self.hidden5)
        self.relu5 = nn.ReLU()
        self.fc6 = nn.Linear(self.hidden5, 1)
        self.apply(weights_init)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.fc3(out)
        out = self.relu3(out)
        out = self.fc4(out)
        out = self.relu4(out)
        out = self.fc5(out)
        out = self.relu5(out)
        out = self.fc6(out)
        return out


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'input_size': 4}]
