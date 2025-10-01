import torch
from torch import nn
import torch.nn.functional as F
import torch.optim


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=
            3, padding=1)
        self.max2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size
            =3, padding=1)
        self.max4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv5 = nn.Conv2d(in_channels=64, out_channels=128,
            kernel_size=3, padding=1)
        self.max6 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv7 = nn.Conv2d(in_channels=128, out_channels=256,
            kernel_size=3, padding=1)
        self.max8 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc9 = nn.Linear(256 * 14 * 14, 4096)
        self.fc10 = nn.Linear(4096, 133)
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.max2(x)
        x = F.relu(self.conv3(x))
        x = self.max4(x)
        x = F.relu(self.conv5(x))
        x = self.max6(x)
        x = F.relu(self.conv7(x))
        x = self.max8(x)
        x = x.view(-1, 256 * 14 * 14)
        x = self.dropout(x)
        x = F.relu(self.fc9(x))
        x = self.dropout(x)
        x = self.fc10(x)
        return x


def get_inputs():
    return [torch.rand([4, 3, 121, 121])]


def get_init_inputs():
    return [[], {}]
