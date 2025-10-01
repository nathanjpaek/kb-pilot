import torch
import torch.nn as nn
import torch.nn.functional as F


class Net1(nn.Module):

    def __init__(self):
        super(Net1, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3)
        self.conv2 = nn.Conv2d(32, 32, 3)
        self.conv3 = nn.Conv2d(32, 64, 3)
        self.conv4 = nn.Conv2d(64, 64, 3)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 5 * 5, 512)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        x = F.elu(self.conv1(x))
        x = F.elu(self.conv2(x))
        x = self.pool1(x)
        x = F.elu(self.conv3(x))
        x = F.elu(self.conv4(x))
        x = self.pool2(x)
        x = x.view(-1, 64 * 5 * 5)
        x = F.elu(self.fc1(x))
        x = self.fc2(x)
        return x

    def linear_layer_ids(self):
        return [8, 10]

    def linear_layer_parameters(self):
        linear1 = torch.cat([x.view(-1) for x in self.fc1.parameters() or
            self.fc2.parameters()])
        return linear1

    def train_order_block_ids(self):
        return [[4, 5], [10, 11], [2, 3], [6, 7], [0, 1], [8, 9]]


def get_inputs():
    return [torch.rand([4, 3, 32, 32])]


def get_init_inputs():
    return [[], {}]
