import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):

    def __init__(self, input_size, out_size, drop_prob=0.5):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.fc2 = nn.Linear(256, out_size)
        self.drop_prob = drop_prob

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.dropout(x, self.drop_prob, self.training)
        x = F.relu(self.fc2(x))
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'input_size': 4, 'out_size': 4}]
