import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):

    def __init__(self, num_classes, n_1, n_2):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(784, n_1)
        self.fc2 = nn.Linear(n_1, n_2)
        self.fc3 = nn.Linear(n_2, num_classes)

    def forward(self, din):
        din = din.view(-1, 28 * 28)
        dout = F.relu(self.fc1(din))
        dout = F.relu(self.fc2(dout))
        return self.fc3(dout)


def get_inputs():
    return [torch.rand([4, 784])]


def get_init_inputs():
    return [[], {'num_classes': 4, 'n_1': 4, 'n_2': 4}]
