import torch
import torch.nn as nn
import torch.nn.functional as F


class DeepLiftRegressor(nn.Module):

    def __init__(self):
        super(DeepLiftRegressor, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=4, out_channels=50, kernel_size=
            (1, 11))
        self.conv2 = nn.Conv2d(in_channels=50, out_channels=50, kernel_size
            =(1, 11))
        self.fc1 = nn.Linear(50, 50)
        self.dropout1 = nn.Dropout2d(p=0.5)
        self.fc2 = nn.Linear(50, 5)

    def forward(self, input):
        x = F.relu(self.conv1(input))
        x = F.relu(self.conv2(x))
        x = torch.mean(x.view(x.size(0), x.size(1), -1), dim=2)
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = self.fc2(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


def get_inputs():
    return [torch.rand([4, 4, 64, 64])]


def get_init_inputs():
    return [[], {}]
