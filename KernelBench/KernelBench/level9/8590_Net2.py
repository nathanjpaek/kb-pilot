import torch
import torch.nn as nn
import torch.nn.functional as F


class Net2(nn.Module):

    def __init__(self):
        super(Net2, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv4 = nn.Conv2d(256, 512, 3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.pool4 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(512 * 2 * 2, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, 512)
        self.fc4 = nn.Linear(512, 1024)
        self.fc5 = nn.Linear(1024, 10)

    def forward(self, x):
        x = F.elu(self.conv1(x))
        x = self.pool1(x)
        x = F.elu(self.conv2(x))
        x = self.pool2(x)
        x = F.elu(self.conv3(x))
        x = self.pool3(x)
        x = F.elu(self.conv4(x))
        x = self.pool4(x)
        x = x.view(-1, 512 * 2 * 2)
        x = F.elu(self.fc1(x))
        x = F.elu(self.fc2(x))
        x = F.elu(self.fc3(x))
        x = F.elu(self.fc4(x))
        x = self.fc5(x)
        return x

    def linear_layer_ids(self):
        return [12, 14, 16]

    def linear_layer_parameters(self):
        linear1 = torch.cat([x.view(-1) for x in self.fc1.parameters() or
            self.fc2.parameters() or self.fc3.parameters() or self.fc4.
            parameters() or self.fc5.parameters()])
        return linear1

    def train_order_block_ids(self):
        return [[14, 15], [4, 5], [2, 3], [8, 9], [16, 17], [12, 13], [6, 7
            ], [0, 1], [10, 11]]


def get_inputs():
    return [torch.rand([4, 3, 64, 64])]


def get_init_inputs():
    return [[], {}]
