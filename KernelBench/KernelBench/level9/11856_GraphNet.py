import torch
import torch.nn as nn


class GraphNet(nn.Module):

    def __init__(self, input_size, n_classes, num_neurons=32):
        super(GraphNet, self).__init__()
        self.fc1 = nn.Linear(input_size, num_neurons)
        self.fc2 = nn.Linear(num_neurons, num_neurons)
        self.fc3 = nn.Linear(num_neurons, n_classes)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.sigmoid(x)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'input_size': 4, 'n_classes': 4}]
