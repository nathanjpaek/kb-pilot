import torch
from torch import nn


class DupCNN(nn.Module):

    def __init__(self, input_shape, output_size, conv_layers, fc_layers):
        super(DupCNN, self).__init__()
        self.input_shape = input_shape
        self.output_size = output_size
        self.conv_layers = conv_layers
        self.fc_layers = fc_layers
        self.conv1 = nn.Conv2d(input_shape[0], 16, kernel_size=3, stride=1,
            padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.fc1 = nn.Linear(256 * 8 * 8, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, output_size)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv4(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv5(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.view(-1, 256 * 8 * 8)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = torch.sigmoid(x)
        return x


def get_inputs():
    return [torch.rand([4, 4, 128, 128])]


def get_init_inputs():
    return [[], {'input_shape': [4, 4], 'output_size': 4, 'conv_layers': 1,
        'fc_layers': 1}]
