import torch
import torch.nn as nn
import torch.nn.functional as F


class cnn_7layer_alt(nn.Module):

    def __init__(self, in_ch, in_dim, width=2, linear_size=128):
        super(cnn_7layer_alt, self).__init__()
        self.conv1 = nn.Conv2d(in_ch, 4 * width, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(4 * width, 4 * width, 4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(4 * width, 8 * width, 3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(8 * width, 8 * width, 4, stride=2, padding=1)
        self.fc1 = nn.Linear(8 * width * (in_dim // 4) * (in_dim // 4),
            linear_size)
        self.fc2 = nn.Linear(linear_size, linear_size)
        self.fc3 = nn.Linear(linear_size, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_ch': 4, 'in_dim': 4}]
