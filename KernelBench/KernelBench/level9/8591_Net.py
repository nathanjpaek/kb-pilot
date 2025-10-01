import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.elu(self.conv1(x)))
        x = self.pool(F.elu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.elu(self.fc1(x))
        x = F.elu(self.fc2(x))
        x = self.fc3(x)
        return x

    def linear_layer_ids(self):
        return [4, 6, 8]

    def linear_layer_parameters(self):
        linear1 = torch.cat([x.view(-1) for x in self.fc1.parameters() or
            self.fc2.parameters() or self.fc3.parameters()])
        return linear1

    def train_order_block_ids(self):
        return [[4, 5], [0, 1], [2, 3], [6, 7], [8, 9]]


def get_inputs():
    return [torch.rand([4, 3, 32, 32])]


def get_init_inputs():
    return [[], {}]
