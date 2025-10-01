import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Net(nn.Module):

    def __init__(self, device):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=24, kernel_size=
            5, padding=0)
        self.conv2 = nn.Conv2d(in_channels=24, out_channels=36, kernel_size
            =4, padding=0)
        self.conv3 = nn.Conv2d(in_channels=36, out_channels=48, kernel_size
            =3, padding=0)
        self.fc1 = nn.Linear(in_features=48, out_features=60)
        self.fc2 = nn.Linear(in_features=60, out_features=10)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.parameters(), lr=0.001, eps=1e-07,
            weight_decay=0.001)
        self.device = device

    def forward(self, x):
        x = self.conv1(x)
        x = F.max_pool2d(x, kernel_size=(3, 3), stride=3)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.max_pool2d(x, kernel_size=(2, 2), stride=2)
        x = F.relu(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = F.avg_pool2d(x, kernel_size=x.size()[2:])
        x = x.view(-1, 48)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x


def get_inputs():
    return [torch.rand([4, 1, 64, 64])]


def get_init_inputs():
    return [[], {'device': 0}]
