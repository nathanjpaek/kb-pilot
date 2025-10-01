import torch
import torch.nn as nn


class Discriminator(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=5, padding=2)
        self.activation1 = nn.Tanh()
        self.maxpool1 = nn.MaxPool2d(kernel_size=(2, 2))
        self.conv2 = nn.Conv2d(64, 128, kernel_size=(5, 5))
        self.activation2 = nn.Tanh()
        self.maxpool2 = nn.MaxPool2d(kernel_size=(2, 2))
        self.fc1 = nn.Linear(128 * 5 * 5, 1024)
        self.activation3 = nn.Tanh()
        self.fc2 = nn.Linear(1024, 512)
        self.activation4 = nn.Tanh()
        self.fc3 = nn.Linear(512, 1)
        self.activation5 = nn.Sigmoid()

    def forward(self, x):
        x = x.view(-1, 1, 28, 28)
        x = self.conv1(x)
        x = self.activation1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.activation2(x)
        x = self.maxpool2(x)
        x = x.view(-1, 128 * 5 * 5)
        x = self.fc1(x)
        x = self.activation3(x)
        x = self.fc2(x)
        x = self.activation4(x)
        x = self.fc3(x)
        x = self.activation5(x)
        return x


def get_inputs():
    return [torch.rand([4, 1, 28, 28])]


def get_init_inputs():
    return [[], {}]
