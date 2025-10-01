import torch
import torch.nn as nn
import torch.nn.functional as functional


class SimpleAtariNet(nn.Module):

    def __init__(self):
        super(SimpleAtariNet, self).__init__()
        self.conv0 = nn.Conv2d(3, 16, 12, stride=(2, 8))
        self.conv1 = nn.Conv2d(16, 32, 8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, 3, stride=1)
        self.lin1 = nn.Linear(1280, 512)
        self.lin2 = nn.Linear(512, 2)

    def forward(self, x):
        x = functional.relu(self.conv0(x))
        x = functional.relu(self.conv1(x))
        x = functional.relu(self.conv2(x))
        x = functional.relu(self.conv3(x))
        x = functional.relu(self.lin1(x.view(-1, 1280)))
        x = self.lin2(x)
        return x


def get_inputs():
    return [torch.rand([4, 3, 576, 576])]


def get_init_inputs():
    return [[], {}]
