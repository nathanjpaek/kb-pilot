import torch
import torch.nn as nn


class ConvNet(nn.Module):

    def __init__(self, img_size):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.relu = nn.ReLU()
        self.padding = nn.ZeroPad2d(1)
        self.fc1 = nn.Linear(4 * img_size * img_size, 128)
        self.fc2 = nn.Linear(128, 84)
        self.fc3 = nn.Linear(84, 4)
        self.flat = nn.Flatten(1)
        self.max_pool = nn.MaxPool2d(2)

    def forward(self, x):
        covlayer1 = self.max_pool(self.relu(self.conv1(self.padding(x))))
        covlayer2 = self.max_pool(self.relu(self.conv2(self.padding(
            covlayer1))))
        covlayer2 = self.flat(covlayer2)
        x = self.relu(self.fc1(covlayer2))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def get_inputs():
    return [torch.rand([4, 3, 4, 4])]


def get_init_inputs():
    return [[], {'img_size': 4}]
