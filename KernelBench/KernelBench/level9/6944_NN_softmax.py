import torch
from torch import nn
import torch.nn.functional as F


class NN_logsoftmax(nn.Module):
    """Build a new class for the network you want to run, returning log 
    softmax"""

    def set_parameters(self, initializers):
        """Set the parameter values obtained from vanilla NN as initializers"""
        with torch.no_grad():
            self.fc1.weight.data = torch.from_numpy(initializers[0].copy())
            self.fc1.bias.data = torch.from_numpy(initializers[1].copy())
            self.fc2.weight.data = torch.from_numpy(initializers[2].copy())
            self.fc2.bias.data = torch.from_numpy(initializers[3].copy())
    """Single layer network with layer_size nodes"""

    def __init__(self, d, layer_size, num_classes):
        super(NN_logsoftmax, self).__init__()
        self.fc1 = nn.Linear(d, layer_size)
        self.fc2 = nn.Linear(layer_size, num_classes)
    """Return the log softmax values for each of the classes"""

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class NN_softmax(NN_logsoftmax):
    """Build a new class for the network you want to run, returning non-log 
    softmax"""
    """Return the softmax values for each of the classes"""

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.softmax(x, dim=1)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'d': 4, 'layer_size': 1, 'num_classes': 4}]
