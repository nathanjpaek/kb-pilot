import torch
from torch import nn


class LongCNN(nn.Module):

    def __init__(self, num_channels, input_shape, name, conv_sizes=[64, 128,
        128, 256], lin_size=512):
        super(LongCNN, self).__init__()
        self.name = name
        self.relu = nn.ReLU(inplace=True)
        self.do1 = nn.Dropout(p=0.25)
        self.do2 = nn.Dropout(p=0.25)
        self.conv1 = nn.Conv2d(num_channels, out_channels=conv_sizes[0],
            kernel_size=6, stride=3)
        self.conv2 = nn.Conv2d(in_channels=conv_sizes[0], out_channels=
            conv_sizes[1], kernel_size=4, stride=2)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(in_channels=conv_sizes[1], out_channels=
            conv_sizes[2], kernel_size=3)
        self.conv4 = nn.Conv2d(in_channels=conv_sizes[2], out_channels=
            conv_sizes[3], kernel_size=3)
        self.pool2 = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(in_features=conv_sizes[-1], out_features=lin_size)
        self.fc2 = nn.Linear(in_features=lin_size, out_features=1)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                    nonlinearity='relu')

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.pool1(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.pool1(x)
        x = self.relu(x)
        x = self.conv4(x)
        x = self.pool1(x)
        x = self.relu(x)
        x = self.pool2(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

    @property
    def save_path(self):
        return self.name + '.chkpt'


def get_inputs():
    return [torch.rand([4, 4, 256, 256])]


def get_init_inputs():
    return [[], {'num_channels': 4, 'input_shape': 4, 'name': 4}]
