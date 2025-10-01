import torch
import torch.nn as nn


class BasicModel_ConvNet_MaxPool3d(nn.Module):
    """Same as above, but with the MaxPool1d replaced
    with a MaxPool3d. This is useful because the MaxPool modules
    behave differently to other modules from the perspective
    of the DeepLift Attributions
    """

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv3d(1, 2, 3)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool3d(2)
        self.conv2 = nn.Conv3d(2, 4, 3)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool3d(2)
        self.fc1 = nn.Linear(4, 8)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(8, 10)
        self.softmax = nn.Softmax(dim=1)
        self.fc1.weight = nn.Parameter(torch.ones(8, 4))
        self.fc2.weight = nn.Parameter(torch.ones(10, 8))

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.pool1(x)
        x = self.relu2(self.conv2(x))
        x = self.pool2(x)
        x = x.view(-1, 4)
        x = self.relu3(self.fc1(x))
        x = self.fc2(x)
        return self.softmax(x)


def get_inputs():
    return [torch.rand([4, 1, 64, 64, 64])]


def get_init_inputs():
    return [[], {}]
