import torch
import torch.nn as nn


class ConvNeuralNetwork(nn.Module):

    def __init__(self, num_classes=3):
        super(ConvNeuralNetwork, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=12, kernel_size=
            3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=12, out_channels=12, kernel_size
            =3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=12, out_channels=24, kernel_size
            =3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=24, out_channels=24, kernel_size
            =3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(in_features=32 * 32 * 24, out_features=num_classes
            )
        self.dropout = nn.Dropout2d(p=0.5)

    def forward(self, input):
        output = self.conv1(input)
        output = self.relu(output)
        output = self.conv2(output)
        output = self.relu(output)
        output = self.pool(output)
        output = self.conv3(output)
        output = self.relu(output)
        output = self.conv4(output)
        output = self.relu(output)
        output = output.view(-1, 32 * 32 * 24)
        output = self.fc1(output)
        return output


def get_inputs():
    return [torch.rand([4, 3, 64, 64])]


def get_init_inputs():
    return [[], {}]
