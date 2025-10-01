import torch
import torch.nn.functional as F


class Complex_nn(torch.nn.Module):

    def __init__(self, dims_in, hidden):
        super(Complex_nn, self).__init__()
        self.fc1 = torch.nn.Linear(dims_in, hidden)
        self.fc2 = torch.nn.Linear(hidden, hidden)
        self.fc3 = torch.nn.Linear(hidden, 2)
        self.fc4 = torch.nn.LogSoftmax()

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'dims_in': 4, 'hidden': 4}]
