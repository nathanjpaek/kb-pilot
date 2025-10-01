import torch
from torch import nn
from torch.nn import functional as F


class CNN(nn.Module):
    """
    conv1, conv2
        two convolution layers.
    fc1, fc2
        two fully connected layers.
    fc3
        output layer
    relu
        activation function for hidden layers
    sigmoid
        activation function for output layer
    """

    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(2, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.fc1 = nn.Linear(256, 200)
        self.fc2 = nn.Linear(200, 10)
        self.fc3 = nn.Linear(10, 2)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), kernel_size=2))
        x = F.relu(F.max_pool2d(self.conv2(x), kernel_size=2))
        x = x.view(-1, 256)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x


def get_inputs():
    return [torch.rand([4, 2, 64, 64])]


def get_init_inputs():
    return [[], {}]
